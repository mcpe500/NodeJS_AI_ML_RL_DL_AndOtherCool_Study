/* User-facing CLI shape (Cactus Needle parity).
 * Today: text tool-calling NOT wired (no BPE, no FC weights).
 * Prints contract + points to working dev checks. Exit 2 = not ready.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s --query TEXT --tools JSON\n\n"
        "Intended (Cactus-style) I/O:\n"
        "  query:  What's the weather in San Francisco?\n"
        "  tools:   [{\"name\":\"get_weather\",\"parameters\":{\"location\":\"string\"}}]\n"
        "  output:  [{\"name\":\"get_weather\",\"arguments\":{\"location\":\"San Francisco\"}}]\n\n"
        "Status: NOT READY in needle.c yet.\n"
        "Missing: BPE tokenizer, function-calling train data, text generate path.\n"
        "What works today (dev):\n"
        "  make -C tests\n"
        "  make smoke MODEL=01-sanity\n"
        "  ./eval/evaluate A data/smoke.bin models/01-sanity/ckpts/sanity.nd\n"
        "  make infer   # token-id greedy only, not natural language\n",
        argv0);
}

int main(int argc, char **argv) {
    const char *query = NULL, *tools = NULL;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--query") && i + 1 < argc) query = argv[++i];
        else if (!strcmp(argv[i], "--tools") && i + 1 < argc) tools = argv[++i];
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        }
    }
    if (!query || !tools) {
        usage(argv[0]);
        return 1;
    }

    printf("=== needle.c chat_demo ===\n");
    printf("INPUT query: %s\n", query);
    printf("INPUT tools: %s\n", tools);
    printf("\n");
    printf("OUTPUT: (not generated)\n");
    printf("  text→tool-call path is NOT wired yet.\n");
    printf("  Cactus reference I/O for this query would look like:\n");
    printf("  [{\"name\":\"get_weather\",\"arguments\":{\"location\":\"San Francisco\"}}]\n");
    printf("\n");
    printf("Why empty:\n");
    printf("  - weights here trained on synthetic token ids (vocab=64 fixture), not English/JSON\n");
    printf("  - no BPE encode/decode\n");
    printf("  - no generate(query, tools) loop\n");
    printf("\n");
    printf("Dev checks that DO work:\n");
    printf("  make -C framework && make -C tests\n");
    printf("  make smoke MODEL=01-sanity\n");
    printf("  make infer\n");
    printf("\n");
    printf("See README.md → \"User try (Cactus-style) vs what exists\".\n");
    return 2;
}
