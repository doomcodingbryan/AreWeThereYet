"""
LLM chat route — only loaded when USE_LLM = True in routes.py.
Adds a POST /api/chat endpoint that implements the RAG pipeline:

  user query → LLM modifies query → IR (search_engine) retrieves results
             → LLM answers from retrieved docs → frontend displays both

Setup:
  1. Add API_KEY=your_key to .env
  2. Set USE_LLM = True in routes.py
"""
import json
import os
import logging
import requests as http_requests
from flask import request, jsonify, Response, stream_with_context

logger = logging.getLogger(__name__)

SPARK_ENDPOINT = "https://4300spark.infosci.cornell.edu/api/chat"


def _spark_call(api_key, messages, stream=False):
    """Call the Cornell Spark LLM endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"messages": messages, "stream": stream}
    resp = http_requests.post(
        SPARK_ENDPOINT,
        headers=headers,
        json=payload,
        stream=stream,
        timeout=60,
    )
    resp.raise_for_status()
    return resp


def _llm_modify_query(api_key, user_message):
    """
    Ask the LLM to transform the user's conversational message into a
    concise IR search query. Returns the modified query string.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a search query assistant for a country recommendation engine. "
                "Given a user's question or request about living in or moving to a country, "
                "output a concise search query of 3-8 keywords that captures the key themes "
                "for an information retrieval system. "
                "Output ONLY the query, nothing else — no explanation, no punctuation, no quotes."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    resp = _spark_call(api_key, messages, stream=False)
    data = resp.json()
    modified = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", user_message)
        .strip()
    )
    return modified or user_message


def _build_context_from_results(top_countries):
    """
    Fetch Reddit posts for the top IR-retrieved countries and format them
    as a RAG context block for the final LLM answer.
    """
    from models import Post, Country as CountryModel

    blocks = []
    for entry in top_countries[:5]:
        name = entry["country"]
        score = entry.get("score", 0)
        posts = (
            Post.query
            .join(Post.countries)
            .filter(CountryModel.name == name)
            .order_by(Post.score.desc())
            .limit(3)
            .all()
        )
        for p in posts:
            blocks.append(
                f"Country: {name} (match score: {score:.2f})\n"
                f"Title: {p.title}\nSubreddit: r/{p.subreddit}\n"
                f"Body: {(p.body or '')[:400]}"
            )

    return "\n\n---\n\n".join(blocks) or "No matching posts found."


def _spark_stream_response(api_key, messages):
    """Stream the final LLM answer, yielding text content chunks."""
    resp = _spark_call(api_key, messages, stream=True)
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if line.startswith("data: "):
            line = line[6:]
        if line.strip() in ("[DONE]", ""):
            break
        try:
            chunk = json.loads(line)
            content = (
                chunk.get("choices", [{}])[0]
                .get("delta", {})
                .get("content", "")
            )
            if content:
                yield content
        except (json.JSONDecodeError, IndexError, KeyError):
            continue


def register_chat_route(app, _json_search, search_engine=None):
    """Register LLM endpoints. Called from routes.py."""

    @app.route("/api/explain", methods=["POST"])
    def explain():
        from models import Post, Country as CountryModel

        data = request.get_json() or {}
        query = (data.get("query") or "").strip()
        countries = data.get("countries", [])

        if not query or not countries:
            return jsonify({"error": "query and countries required"}), 400

        api_key = os.getenv("API_KEY")
        if not api_key:
            return jsonify({"error": "API_KEY not set"}), 500

        def generate():
            for entry in countries[:5]:
                country_name = entry.get("country", "")
                score = entry.get("score", 0)

                posts = (
                    Post.query
                    .join(Post.countries)
                    .filter(CountryModel.name == country_name)
                    .order_by(Post.score.desc())
                    .limit(3)
                    .all()
                )

                if not posts:
                    continue

                post_context = "\n\n".join(
                    f"Title: {p.title}\nBody: {(p.body or '')[:400]}"
                    for p in posts
                )

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a country relocation expert. Given a user's search query and Reddit posts "
                            "about a country, write exactly 2-3 sentences explaining why this country matches "
                            "the query. Be specific — cite themes or details from the posts. "
                            "Output only the explanation, no headers or labels."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Query: \"{query}\"\n"
                            f"Country: {country_name} ({round(score * 100)}% match)\n\n"
                            f"Reddit posts:\n{post_context}"
                        ),
                    },
                ]

                try:
                    resp = _spark_call(api_key, messages, stream=False)
                    explanation = (
                        resp.json()
                        .get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
                    if explanation:
                        yield f"data: {json.dumps({'country': country_name, 'explanation': explanation})}\n\n"
                except Exception as e:
                    logger.error(f"Explanation error for {country_name}: {e}")

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        api_key = os.getenv("API_KEY")
        if not api_key:
            return jsonify({"error": "API_KEY not set — add it to your .env file"}), 500

        def generate():
            # Step 1: LLM modifies the user query for IR
            try:
                modified_query = _llm_modify_query(api_key, user_message)
            except Exception as e:
                logger.error(f"Query modification error: {e}")
                modified_query = user_message

            logger.info(f"Modified query: {modified_query!r}")

            # Emit modified query so frontend can update the search bar + IR results
            yield f"data: {json.dumps({'search_term': modified_query})}\n\n"

            # Step 2: Run IR on the modified query
            ir_results = []
            if search_engine is not None:
                try:
                    ir_results = search_engine.search(modified_query, top_k=10)
                except Exception as e:
                    logger.error(f"IR search error: {e}")

            # Emit IR results so frontend can render country cards immediately
            yield f"data: {json.dumps({'ir_results': ir_results})}\n\n"

            # Step 3: Build RAG context from retrieved documents
            context_text = _build_context_from_results(ir_results)
            country_list = ", ".join(r["country"] for r in ir_results[:5])

            # Step 4: Stream final LLM answer grounded in the IR results
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a country relocation expert. Using only the Reddit posts provided, "
                        "write a concise 3-4 sentence answer to the user's question. "
                        "Be specific — cite details from the posts. "
                        "Mention which countries were retrieved and why they match."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User question: \"{user_message}\"\n\n"
                        f"IR search query used: \"{modified_query}\"\n\n"
                        f"Top retrieved countries: {country_list}\n\n"
                        f"Reddit posts:\n\n{context_text}"
                    ),
                },
            ]

            try:
                for chunk in _spark_stream_response(api_key, messages):
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': 'Streaming error occurred'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
