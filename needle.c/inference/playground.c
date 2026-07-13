/* Playground HTTP server — bind 0.0.0.0 (Termux-friendly).
 * Live path: encode(query+tools) → greedy decode → detok JSON.
 * Usage: ./inference/playground [port] [ckpt] [vocab] [merges]
 *   defaults: 7860, models/01-sanity/ckpts/sanity_fc.nd, tokenizer/vocab+merges
 */
#include "needle.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <errno.h>

#define HOST "0.0.0.0"
#define DEF_PORT 7860
#define BUF 65536
#define MAX_SRC 128
#define MAX_TGT 64

static nd_module *g_model;
static nd_bpe g_bpe;
static int g_ready;

static const char *PAGE =
"<!DOCTYPE html><html><head><meta charset=utf-8>"
"<meta name=viewport content=\"width=device-width,initial-scale=1\">"
"<title>needle.c playground</title>"
"<style>"
"body{font-family:system-ui,sans-serif;max-width:720px;margin:1.5rem auto;padding:0 1rem;background:#0f1115;color:#e8eaed}"
"h1{font-size:1.25rem} label{display:block;margin:.75rem 0 .25rem;color:#9aa0a6}"
"textarea,input{width:100%;box-sizing:border-box;background:#1a1d24;color:#e8eaed;border:1px solid #333;border-radius:8px;padding:.6rem;font:inherit}"
"textarea{min-height:5rem} button{margin-top:1rem;padding:.6rem 1.2rem;border:0;border-radius:8px;background:#8ab4f8;color:#0f1115;font-weight:600;cursor:pointer}"
"pre{background:#1a1d24;border:1px solid #333;border-radius:8px;padding:1rem;white-space:pre-wrap;word-break:break-word}"
".warn{color:#fdd663;font-size:.9rem} .ok{color:#81c995} a{color:#8ab4f8}"
"</style></head><body>"
"<h1>needle.c playground</h1>"
"<p class=ok>Live model path: encode → greedy → detok. Output is model string (may underfit).</p>"
"<form method=POST action=/generate>"
"<label>query</label>"
"<textarea name=query>What's the weather in San Francisco?</textarea>"
"<label>tools (JSON)</label>"
"<textarea name=tools>[{\"name\":\"get_weather\",\"parameters\":{\"location\":\"string\"}}]</textarea>"
"<button type=submit>Generate</button>"
"</form>"
"<p style=margin-top:2rem;font-size:.85rem;color:#9aa0a6>"
"Bind: 0.0.0.0 · <code>make fc</code> then <code>make playground</code>"
"</p></body></html>";

static void url_decode(char *s) {
    char *o = s;
    for (char *p = s; *p; ++p) {
        if (*p == '+' ) { *o++ = ' '; }
        else if (*p == '%' && isxdigit((unsigned char)p[1]) && isxdigit((unsigned char)p[2])) {
            char hex[3] = {p[1], p[2], 0};
            *o++ = (char)strtol(hex, NULL, 16);
            p += 2;
        } else *o++ = *p;
    }
    *o = 0;
}

static int form_get(const char *body, const char *key, char *out, size_t outn) {
    size_t klen = strlen(key);
    const char *p = body;
    while (p && *p) {
        if (strncmp(p, key, klen) == 0 && p[klen] == '=') {
            p += klen + 1;
            size_t i = 0;
            while (*p && *p != '&' && i + 1 < outn) out[i++] = *p++;
            out[i] = 0;
            url_decode(out);
            return 0;
        }
        p = strchr(p, '&');
        if (p) ++p;
    }
    out[0] = 0;
    return -1;
}

static void send_all(int fd, const char *s, size_t n) {
    while (n) {
        ssize_t w = write(fd, s, n);
        if (w <= 0) break;
        s += (size_t)w; n -= (size_t)w;
    }
}

static void respond(int fd, int code, const char *ctype, const char *body) {
    char hdr[256];
    size_t blen = strlen(body);
    int n = snprintf(hdr, sizeof hdr,
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
        "Connection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
        code, code == 200 ? "OK" : "Bad Request", ctype, blen);
    send_all(fd, hdr, (size_t)n);
    send_all(fd, body, blen);
}

/* Greedy generate into out (size outn). Returns 0 ok. */
static int live_generate(const char *query, const char *tools, char *out, size_t outn) {
    if (!g_ready || !g_model) {
        snprintf(out, outn, "[]");
        return -1;
    }
    char src_text[4096];
    snprintf(src_text, sizeof src_text, "Query: %s\nTools: %s", query, tools);
    int src_ids[MAX_SRC];
    int ns = nd_bpe_encode(&g_bpe, src_text, src_ids, MAX_SRC, 1);
    if (ns <= 0) { snprintf(out, outn, "[]"); return -1; }

    int src_pad[MAX_SRC];
    for (int i = 0; i < MAX_SRC; ++i)
        src_pad[i] = i < ns ? src_ids[i] : g_bpe.pad_id;

    int tgt_ids[MAX_TGT];
    int nt = 1;
    tgt_ids[0] = g_bpe.bos_id;

    for (int step = 0; step < MAX_TGT - 1 && nt < MAX_TGT; ++step) {
        int tgt_pad[MAX_TGT];
        for (int i = 0; i < MAX_TGT; ++i)
            tgt_pad[i] = i < nt ? tgt_ids[i] : g_bpe.pad_id;

        int ssh[2] = {1, MAX_SRC}, tsh[2] = {1, MAX_TGT};
        nd_tensor *src = nd_zeros(ssh, 2, false);
        nd_tensor *tgt = nd_zeros(tsh, 2, false);
        for (int i = 0; i < MAX_SRC; ++i) src->data[i] = (float)src_pad[i];
        for (int i = 0; i < MAX_TGT; ++i) tgt->data[i] = (float)tgt_pad[i];

        nd_tensor *logits = g_model->forward2(g_model, src, tgt);
        int V = logits->shape[2];
        int last = nt - 1;
        float *row = logits->data + last * V;
        int best = 0; float bv = row[0];
        for (int v = 1; v < V; ++v) if (row[v] > bv) { bv = row[v]; best = v; }
        nd_tensor_free(logits);
        nd_tensor_free(src); nd_tensor_free(tgt);

        if (best == g_bpe.eos_id) break;
        tgt_ids[nt++] = best;
    }
    nd_bpe_decode(&g_bpe, tgt_ids + 1, nt - 1, out, (int)outn);
    return 0;
}

static void handle(int cfd) {
    char req[BUF];
    ssize_t n = read(cfd, req, sizeof req - 1);
    if (n <= 0) { close(cfd); return; }
    req[n] = 0;

    if (strncmp(req, "GET / ", 6) == 0 || strncmp(req, "GET /index", 10) == 0) {
        respond(cfd, 200, "text/html; charset=utf-8", PAGE);
        close(cfd);
        return;
    }

    if (strncmp(req, "POST /generate", 14) == 0) {
        char *body = strstr(req, "\r\n\r\n");
        if (!body) { respond(cfd, 400, "text/plain", "no body"); close(cfd); return; }
        body += 4;
        char query[4096], tools[8192], model_out[4096], html[16384];
        form_get(body, "query", query, sizeof query);
        form_get(body, "tools", tools, sizeof tools);
        for (char *p = query; *p; ++p) if (*p == '<') *p = '[';
        for (char *p = tools; *p; ++p) if (*p == '<') *p = '[';

        live_generate(query, tools[0] ? tools : "[]", model_out, sizeof model_out);
        for (char *p = model_out; *p; ++p) if (*p == '<') *p = '[';

        snprintf(html, sizeof html,
            "<!DOCTYPE html><html><head><meta charset=utf-8>"
            "<meta name=viewport content=\"width=device-width,initial-scale=1\">"
            "<title>result — needle.c</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:720px;margin:1.5rem auto;padding:0 1rem;"
            "background:#0f1115;color:#e8eaed} pre{background:#1a1d24;border:1px solid #333;border-radius:8px;"
            "padding:1rem;white-space:pre-wrap;word-break:break-word} .ok{color:#81c995} a{color:#8ab4f8}</style>"
            "</head><body><h1>result</h1>"
            "<p class=ok>Live model output (greedy, no hardcode).</p>"
            "<h3>INPUT query</h3><pre>%s</pre>"
            "<h3>INPUT tools</h3><pre>%s</pre>"
            "<h3>OUTPUT (model)</h3>"
            "<pre>%s</pre>"
            "<p><a href=/>← back</a></p>"
            "</body></html>",
            query[0] ? query : "(empty)",
            tools[0] ? tools : "(empty)",
            model_out[0] ? model_out : "(empty model string)");
        respond(cfd, 200, "text/html; charset=utf-8", html);
        close(cfd);
        return;
    }

    respond(cfd, 404, "text/plain", "not found\n");
    close(cfd);
}

int main(int argc, char **argv) {
    int port = DEF_PORT;
    const char *ckpt = "models/01-sanity/ckpts/sanity_fc.nd";
    const char *vocab = "tokenizer/vocab.json";
    const char *merges = "tokenizer/merges.txt";
    if (argc > 1) port = atoi(argv[1]);
    if (argc > 2) ckpt = argv[2];
    if (argc > 3) vocab = argv[3];
    if (argc > 4) merges = argv[4];
    if (port <= 0 || port > 65535) port = DEF_PORT;

    g_ready = 0;
    if (nd_bpe_load(&g_bpe, vocab, merges) != 0) {
        fprintf(stderr, "warn: BPE load fail %s — generate returns []\n", vocab);
    } else {
        nd_seed(1);
        nd_needle_config cfg = nd_cfg_sanity();
        cfg.vocab = g_bpe.vocab_size;
        cfg.max_len = MAX_SRC;
        g_model = nd_needle_create(&cfg);
        if (nd_checkpoint_load(ckpt, g_model) != 0) {
            fprintf(stderr, "warn: ckpt load fail %s — random weights\n", ckpt);
        } else {
            g_ready = 1;
            printf("loaded ckpt %s vocab=%d\n", ckpt, g_bpe.vocab_size);
        }
    }

    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) { perror("socket"); return 1; }
    int one = 1;
    setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = inet_addr(HOST);

    if (bind(sfd, (struct sockaddr *)&addr, sizeof addr) < 0) {
        perror("bind");
        fprintf(stderr, "port %d in use? try: ./inference/playground 8765\n", port);
        return 1;
    }
    if (listen(sfd, 8) < 0) { perror("listen"); return 1; }

    printf("needle.c playground\n");
    printf("  listen %s:%d\n", HOST, port);
    printf("  open   http://127.0.0.1:%d/\n", port);
    printf("  status %s\n", g_ready ? "LIVE generate()" : "no ckpt — empty output");
    printf("  stop   Ctrl+C\n");
    fflush(stdout);

    for (;;) {
        struct sockaddr_in cli;
        socklen_t cl = sizeof cli;
        int cfd = accept(sfd, (struct sockaddr *)&cli, &cl);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            break;
        }
        handle(cfd);
    }
    close(sfd);
    if (g_model) nd_module_free(g_model);
    return 0;
}
