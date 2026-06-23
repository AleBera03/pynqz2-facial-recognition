#ifndef DB_H
#define DB_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#define DATABASE_PATH "face_database.csv"
#define EMBEDDING_DIM 2048

using namespace std;


struct identity {
    int           id;
    string        name;
    vector<float> embedding;   // 2048-D, L2-normalized
};


// DB is a vector<identity> loaded on RAM
// call write() to persist dato to DB CSV file
class db {
private:
    vector<identity> data_;
    bool             loaded_;

public:
    db();

    // load CSV in memory
    bool load();

    // persistely overwrite CSV file
    bool write();

    // check if id row exists
    bool exists(int id) const;

    // serialize from file an identity
    // nullptr if 'id' does not exist
    const identity* find(int id) const;

    // add or overwrite a row for identity.id.
    // call write() after this in order to persist data
    void upsert(const identity& ident);

    // remove id from DB, noop if does exists
    void remove(int id);

    // insert a new row WITHOUT removing existing ones with same id.
    // used for multi-sample enrollment: each photo is a separate sample.
    void insert(const identity& ident);

    // count how many rows exist with that id
    int count(int id) const;

    // complete access to vector
    const vector<identity>& all() const { return data_; }

    bool is_loaded() const { return loaded_; }
};

#endif
