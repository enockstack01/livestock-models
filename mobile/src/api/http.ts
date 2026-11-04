// Real network adapter for the core's Http interface: fetch + AbortController timeouts, errors mapped to NetworkError / HttpError.
import { HttpError, NetworkError, type FormFile, type Http, type Req } from '../core/http';

async function send(url: string, init: RequestInit, o: Req): Promise<any> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), o.timeoutMs);
  let res: Response;
  try {
    res = await fetch(url, { ...init, signal: ctl.signal, headers: { ...(init.headers as Record<string, string>), ...(o.token ? { Authorization: `Bearer ${o.token}` } : {}) } });
  } catch {
    throw new NetworkError();                                           // no connection, DNS failure, TLS failure or our timeout
  } finally {
    clearTimeout(timer);
  }
  let body: any = null;
  try { body = await res.json(); } catch { /* non-JSON error page */ }
  if (!res.ok) throw new HttpError(res.status, body?.error?.message ?? `HTTP ${res.status}`, body);
  return body;
}

export const http: Http = {
  get: (url, o) => send(url, { method: 'GET' }, o),
  postJson: (url, body, o) => send(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }, o),
  postForm: (url, fields, file: FormFile, o) => {
    const fd = new FormData();
    for (const [k, v] of Object.entries(fields)) fd.append(k, v);
    fd.append('image', file as unknown as Blob);                        // React Native FormData accepts {uri, name, type}; the browser Blob typing does not know it
    return send(url, { method: 'POST', body: fd }, o);                  // no Content-Type header: fetch sets the multipart boundary
  },
};
