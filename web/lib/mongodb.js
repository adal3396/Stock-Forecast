import { MongoClient } from "mongodb";

const DB_NAME = "stock_forecast";

let clientPromise;

function getMongoUri() {
  const uri = process.env.MONGODB_URI;
  if (!uri) {
    throw new Error(
      "MONGODB_URI is not set. Add it in Vercel Project Settings → Environment Variables (and locally in .env.local)."
    );
  }
  return uri;
}

function getClientPromise() {
  if (clientPromise) return clientPromise;
  const uri = getMongoUri();
  const client = new MongoClient(uri);
  if (process.env.NODE_ENV === "development") {
    if (!global._mongoClientPromise) {
      global._mongoClientPromise = client.connect();
    }
    clientPromise = global._mongoClientPromise;
  } else {
    clientPromise = client.connect();
  }
  return clientPromise;
}

/** @returns {Promise<import('mongodb').Db>} */
export async function getDB() {
  const client = await getClientPromise();
  return client.db(DB_NAME);
}
