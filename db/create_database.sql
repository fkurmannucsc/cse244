DROP TABLE IF EXISTS Actions;
DROP TABLE IF EXISTS Tokens;
DROP TABLE IF EXISTS Users;
DROP TABLE IF EXISTS Labels;
DROP TABLE IF EXISTS Tags;
DROP TABLE IF EXISTS AlternateNames;
DROP TABLE IF EXISTS Sharks;
DROP TABLE IF EXISTS Media;

CREATE TABLE Media (
    id INTEGER PRIMARY KEY,
    hash TEXT NOT NULL UNIQUE,
    name TEXT UNIQUE,
    filename TEXT UNIQUE,
    source INTEGER REFERENCES Media(id),
    metadata TEXT,
    sex TEXT,
    length REAL,
    location TEXT,
    time INTEGER,
    type TEXT,
    quality TEXT
    -- TODO who, what, copyright?
);

CREATE TABLE Sharks (
    id INTEGER PRIMARY KEY,
    sex TEXT,
    length REAL,
    media INTEGER REFERENCES Media(id),
    source INTEGER REFERENCES Sharks(id)
);

CREATE TABLE AlternateNames (
    id INTEGER PRIMARY KEY,
    shark INTEGER REFERENCES Sharks(id) ON DELETE CASCADE NOT NULL,
    name TEXT NOT NULL,
    type TEXT,
    time integer
);

CREATE TABLE Tags (
    id INTEGER PRIMARY KEY,
    shark INTEGER REFERENCES Sharks(id) ON DELETE CASCADE NOT NULL,
    name TEXT NOT NULL,
    type TEXT,
    time integer
);

CREATE TABLE Labels (
    id INTEGER PRIMARY KEY,
    shark INTEGER REFERENCES Sharks(id) ON DELETE CASCADE NOT NULL,
    name TEXT NOT NULL,
    type TEXT,
    time integer
);

CREATE TABLE Users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    username TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL,
    password TEXT NOT NULL,
    salt TEXT NOT NULL
);

CREATE TABLE Tokens (
    id INTEGER PRIMARY KEY,
    token TEXT NOT NULL,
    user INTEGER REFERENCES Users(id) ON DELETE CASCADE NOT NULL
);

CREATE TABLE Actions (
    id INTEGER PRIMARY KEY,
    original INTEGER,
    final INTEGER,
    action TEXT NOT NULL,
    object TEXT NOT NULL,
    author INTEGER REFERENCES Users (id) NOT NULL,
    time integer NOT NULL,
    notes TEXT
)