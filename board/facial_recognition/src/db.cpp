#include "db.h"


db::db() : loaded_(false) {}


// load: read CSV and fill data_
// row format: id,name,e0,e1,...,e255
bool db::load()
{
    data_.clear();

    ifstream f(DATABASE_PATH);
    if (!f.is_open()) {
        // create an empty file
        ofstream f_new(DATABASE_PATH);
        if (!f_new.is_open()) {
            cerr << "[DB] Cannot create " << DATABASE_PATH << endl;
            return false;
        }
        cout << "[DB] Created empty database: " << DATABASE_PATH << endl;
        loaded_ = true;
        return true;
    }

    string line;
    int line_num = 0;
    while (getline(f, line)) {
        line_num++;
        if (line.empty() || line[0] == '#') continue;

        istringstream ss(line);
        string token;
        identity ident;

        // id
        if (!getline(ss, token, ',')) goto malformed;
        try { ident.id = stoi(token); } catch (...) { goto malformed; }

        // nome
        if (!getline(ss, token, ',')) goto malformed;
        ident.name = token;

        // embedding
        ident.embedding.reserve(EMBEDDING_DIM);
        while (getline(ss, token, ','))
            try { ident.embedding.push_back(stof(token)); } catch (...) { goto malformed; }

        if ((int)ident.embedding.size() != EMBEDDING_DIM) goto malformed;

        data_.push_back(move(ident));
        continue;

    malformed:
        cerr << "[DB] Malformed line " << line_num << ", skipped" << endl;
    }

    loaded_ = true;
    cout << "[DB] Readed " << data_.size() << " rows from " << DATABASE_PATH << endl;
    return true;
}


bool db::write()
{
    ofstream f(DATABASE_PATH, ios::out | ios::trunc);
    if (!f.is_open()) {
        cerr << "[DB] Cannot write " << DATABASE_PATH << endl;
        return false;
    }

    for (const auto& ident : data_) {
        f << ident.id << "," << ident.name;
        for (float v : ident.embedding)
            f << "," << v;
        f << "\n";
    }

    cout << "[DB] Written " << data_.size() << " rows" << endl;
    return true;
}


bool db::exists(int id) const
{
    for (const auto& ident : data_)
        if (ident.id == id) return true;
    return false;
}

const identity* db::find(int id) const
{
    for (const auto& ident : data_)
        if (ident.id == id) return &ident;
    return nullptr;
}


void db::upsert(const identity& ident)
{
    for (auto& existing : data_) {
        if (existing.id == ident.id) {
            existing = ident;   // overwriting
            cout << "[DB] Updated id=" << ident.id << " name=" << ident.name << endl;
            return;
        }
    }
    // not found: new insertion
    data_.push_back(ident);
    cout << "[DB] Inserted id=" << ident.id << " name=" << ident.name << endl;
}


void db::remove(int id)
{
    int removed = 0;
    auto it = data_.begin();
    while (it != data_.end()) {
        if (it->id == id) {
            it = data_.erase(it);
            removed++;
        } else {
            ++it;
        }
    }
    if (removed > 0)
        cout << "[DB] Removed " << removed << " sample(s) for id=" << id << endl;
    else
        cerr << "[DB] Remove: id=" << id << " not found" << endl;
}

void db::insert(const identity& ident)
{
    data_.push_back(ident);
    cout << "[DB] Inserted sample id=" << ident.id
         << " name=" << ident.name
         << " (total: " << count(ident.id) << " samples)" << endl;
}

int db::count(int id) const
{
    int n = 0;
    for (const auto& ident : data_)
        if (ident.id == id) n++;
    return n;
}
