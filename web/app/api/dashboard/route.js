import { NextResponse } from "next/server";
import { createServiceClient } from "@/lib/supabase-server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  try {
    const supabase = createServiceClient();
    const { data: rows, error } = await supabase
      .from("dashboard_documents")
      .select("doc_type, payload");

    if (error) throw new Error(error.message);

    const data = {};
    for (const row of rows ?? []) {
      const t = row.doc_type;
      if (!t || !row.payload) continue;
      data[t] =
        typeof row.payload === "object" && row.payload !== null
          ? row.payload
          : {};
    }

    return NextResponse.json({ ok: true, data });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}
