# Developer Checklist: MongoDB v8.2 Local Vector Search

1. **Confirm binary versions**
   - `mongod --version` and `mongot --version` must both report 8.2.2 or newer.
   - `mongosh --eval "db.runCommand({ buildInfo: 1 })" | grep version` to validate the running server matches the binaries on disk.

2. **Verify `mongod.conf` search block**
   - Ensure `/etc/mongod.conf` (or your custom config) contains:
     ```yaml
     search:
       enabled: true
       tutorial: false
       path: /var/lib/mongo/mongot
     ```
   - Restart `mongod` after editing so the embedded `mongot` process is spawned.

3. **Check mongot health**
   - `ps aux | grep mongot` should show the co-resident process.
   - From `mongosh`: `db.adminCommand({ ping: 1, search: "local" })` should return `{ ok: 1 }`.

4. **Feature flag sanity**
   - Run `db.adminCommand({ getParameter: 1, featureFlagSearchIndexes: 1 })` and `featureFlagRankFusion` to ensure `value: { enabled: true }`.
   - `db.adminCommand({ setParameter: 1, featureFlagRankFusion: true })` requires admin role if not already enabled.

5. **Storage path permissions**
   - Confirm the directory referenced by `search.path` is writable by the `mongod` user and has sufficient disk space for vector indexes.

6. **Network bindings**
   - If clients connect remotely, verify `net.bindIp` includes the client host and `security.authorization` is enabled with appropriate users.

7. **Index validation commands**
   - `db.getCollection("documents").getSearchIndexes()` should list the vector index created by `rag_mongo.py`.
   - `db.getCollection("documents").getIndexes()` must show the text index referenced by `--text-index` (default `rag_text_index`).

8. **Sanity query**
   - Execute a lightweight pipeline before running the RAG script:
     ```javascript
     db.documents.aggregate([
       {
         $vectorSearch: {
           index: "rag_vector_index",
           path: "embedding",
           queryVector: Array(384).fill(0),
           numCandidates: 5,
           limit: 1
         }
       }
     ])
     ```
   - Successful execution confirms that the embedded `mongot` process is accepting vector workloads locally.
