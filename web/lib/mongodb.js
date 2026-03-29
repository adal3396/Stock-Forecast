import { MongoClient, ServerApiVersion } from "mongodb";

const DB_NAME = "stock_forecast";

let clientPromise;

/** Atlas + Vercel: force IPv4 — avoids TLS "alert internal error" when IPv6/SRV path fails. */
const CLIENT_OPTIONS = {
  serverApi: {
    version: ServerApiVersion.v1,
    strict: true,
    deprecationErrors: true,
  },
  // Serverless: small pool limits open connections on Atlas
  maxPoolSize: 5,
  minPoolSize: 0,
  serverSelectionTimeoutMS: 15_000,
  connectTimeoutMS: 15_000,
  socketTimeoutMS: 30_000,
  family: 4,
};

function getMongoUri() {
  const uri = process.env.MONGODB_URI?.trim();
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
  const client = new MongoClient(uri, CLIENT_OPTIONS);
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
