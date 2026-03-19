import json
import os
from dotenv import load_dotenv
from flask import Flask

load_dotenv()
from flask_cors import CORS
from models import db, Post, Country
from routes import register_routes

# src/ directory and project root (one level up)
current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_directory)

# Serve React build files from <project_root>/frontend/dist
app = Flask(__name__,
    static_folder=os.path.join(project_root, 'frontend', 'dist'),
    static_url_path='')
CORS(app)

# Configure SQLite database - using 3 slashes for relative path
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database with app
db.init_app(app)

# Register routes
register_routes(app)

def init_db():
    with app.app_context():
        db.create_all()

        if Post.query.count() == 0:
            json_file_path = os.path.join(project_root, 'data', 'reddit_sample.json')

            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                # Cache countries to avoid repeated DB hits
                country_cache = {
                    c.name: c for c in Country.query.all()
                }

                for post_id, post_data in data.items():
                    post = Post(
                        id=post_id,
                        subreddit=post_data.get('subreddit'),
                        title=post_data.get('title', ''),
                        body=post_data.get('body'),
                        full_text=post_data.get('full_text'),
                        score=post_data.get('score'),
                        upvote_ratio=post_data.get('upvote_ratio'),
                        num_comments=post_data.get('num_comments'),
                        created_utc=post_data.get('created_utc'),
                        url=post_data.get('url'),
                        flair=post_data.get('flair'),
                        num_countries=post_data.get('num_countries'),
                        body_length=post_data.get('body_length'),
                        has_country=(post_data.get('has_country') == "True")
                    )

                    # Handle countries (many-to-many)
                    for country_name in post_data.get('countries', []):
                        if country_name in country_cache:
                            country = country_cache[country_name]
                        else:
                            country = Country(name=country_name)
                            db.session.add(country)
                            country_cache[country_name] = country

                        post.countries.append(country)

                    db.session.add(post)

            db.session.commit()
            print("Database initialized with normalized countries")

init_db()

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
