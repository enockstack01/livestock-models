// The only thing the core needs from the network. The real implementation (src/api/http.ts) uses fetch + AbortController;
// tests inject a fake, so every failure mode (offline, timeout, 401, 422, 503) can be simulated deterministically.
export class NetworkError extends Error {
  constructor(message = 'Cannot reach the server') { super(message); this.name = 'NetworkError'; }
}
export class HttpError extends Error {
  status: number; body: unknown;
  constructor(status: number, message: string, body: unknown = null) { super(message); this.name = 'HttpError'; this.status = status; this.body = body; }
}
export type Req = { token?: string; timeoutMs: number };
export type FormFile = { uri: string; name: string; type: string };

/** Each method resolves with the parsed JSON body, or throws NetworkError (no connection / timeout) or HttpError (non-2xx). */
export interface Http {
  get(url: string, o: Req): Promise<any>;
  postJson(url: string, body: unknown, o: Req): Promise<any>;
  postForm(url: string, fields: Record<string, string>, file: FormFile, o: Req): Promise<any>;
}

export const TIMEOUTS = { probeMs: 8000, detectMs: 30_000, reportMs: 90_000 };   // 8 s connectivity probe: the platform spec
