CREATE DATABASE IF NOT EXISTS fireeye;

USE fireeye;

CREATE TABLE IF NOT EXISTS users (
    id int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    name varchar(255) NOT NULL UNIQUE,
    password varchar(255) NOT NULL,
    created_at timestamp DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted_at timestamp
);

CREATE TABLE IF NOT EXISTS records (
    id int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    uid int(11),
    path varchar(255) NOT NULL,
    request_time int(10),
    finished_time int(10),
    result TINYINT NOT NULL,
    created_at timestamp DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted_at timestamp,
    CONSTRAINT fk_records_users FOREIGN KEY(uid) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS polenums (
    id int(11) PRIMARY KEY NOT NULL AUTO_INCREMENT,
    uid int(11),
    path varchar(255) NOT NULL,
    request_time int(10),
    finished_time int(10),
    left_top_x int(10),
    left_top_y int(10),
    right_bottom_x int (10),
    right_bottom_y int (10),
    name varchar(255),
    created_at timestamp DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted_at timestamp,
    CONSTRAINT fk_records_users FOREIGN KEY(uid) REFERENCES users(id)
);