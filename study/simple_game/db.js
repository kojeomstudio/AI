console.log('db.js loaded');

const DB_NAME = 'roguelikeGame';
const DB_VERSION = 1;
let db;

function initDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);

        request.onerror = (event) => {
            console.error('Database error:', event.target.error);
            reject(event.target.error);
        };

        request.onsuccess = (event) => {
            db = event.target.result;
            resolve(db);
        };

        request.onupgradeneeded = (event) => {
            db = event.target.result;
            const objectStore = db.createObjectStore('users', { keyPath: 'username' });
            objectStore.createIndex('password', 'password', { unique: false });
        };
    });
}

function checkCredentials(username, password) {
    return new Promise((resolve, reject) => {
        if (!db) {
            reject(new Error('Database not initialized'));
            return;
        }
        
        const transaction = db.transaction(['users'], 'readonly');
        const objectStore = transaction.objectStore('users');
        const request = objectStore.get(username);

        request.onerror = (event) => {
            console.error('Error fetching user:', event.target.error);
            reject(event.target.error);
        };

        request.onsuccess = (event) => {
            const user = event.target.result;
            if (user) {
                if (user.password === password) {
                    resolve(true);
                } else {
                    resolve(false);
                }
            } else {
                createUser(username, password).then(() => resolve(true));
            }
        };
    });
}

function createUser(username, password) {
    return new Promise((resolve, reject) => {
        const transaction = db.transaction(['users'], 'readwrite');
        const objectStore = transaction.objectStore('users');
        const request = objectStore.add({ username, password });

        request.onerror = (event) => {
            console.error('Error creating user:', event.target.error);
            reject(event.target.error);
        };

        request.onsuccess = (event) => {
            resolve();
        };
    });
}
