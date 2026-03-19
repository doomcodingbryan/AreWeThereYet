from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# Association table (many-to-many)
post_countries = db.Table(
    'post_countries',
    db.Column('post_id', db.String, db.ForeignKey('posts.id'), primary_key=True),
    db.Column('country_id', db.Integer, db.ForeignKey('countries.id'), primary_key=True)
)

class Post(db.Model):
    __tablename__ = 'posts'

    id = db.Column(db.String, primary_key=True)

    subreddit = db.Column(db.String(64))
    title = db.Column(db.String(512), nullable=False)
    body = db.Column(db.Text)
    full_text = db.Column(db.Text)

    score = db.Column(db.Integer)
    upvote_ratio = db.Column(db.Float)
    num_comments = db.Column(db.Integer)

    created_utc = db.Column(db.String(32))
    url = db.Column(db.String(512))
    flair = db.Column(db.String(128))

    num_countries = db.Column(db.Integer)
    body_length = db.Column(db.Integer)
    has_country = db.Column(db.Boolean)

    # Relationship
    countries = db.relationship(
        'Country',
        secondary=post_countries,
        backref=db.backref('posts', lazy='dynamic')
    )

    def __repr__(self):
        return f'<Post {self.id}: {self.title}>'

class Country(db.Model):
    __tablename__ = 'countries'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, nullable=False)

    def __repr__(self):
        return f'<Country {self.name}>'