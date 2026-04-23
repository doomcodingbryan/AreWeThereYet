import os
os.chdir('src')
from models import db, Post
from app import app
with app.app_context():
    print('Posts count:', Post.query.count())
    if Post.query.count() > 0:
        post = Post.query.first()
        print('Sample post:', post.title[:50])
        print('Countries:', [c.name for c in post.countries])