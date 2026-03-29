import { NextResponse } from "next/server";
import { getDB } from "@/lib/mongodb";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  try {
    const db = await getDB();
    const docs = await db.collection("dashboard").find({}).toArray();
    const data = {};
    for (const doc of docs) {
      const t = doc.type;
      if (!t) continue;
      const rest = { ...doc };
      delete rest._id;
      delete rest.type;
      data[t] = rest;
    }
    return NextResponse.json({ ok: true, data });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}
