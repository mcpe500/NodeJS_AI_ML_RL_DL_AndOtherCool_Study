.PHONY: framework tests smoke train eval infer clean

framework:
	$(MAKE) -C framework

tests: framework
	$(MAKE) -C tests

MODEL ?= 01-sanity
WHICH ?= A

smoke: framework
	$(MAKE) -C models/$(MODEL) smoke

train: framework
	$(MAKE) -C models/$(MODEL) train

eval: framework
	$(MAKE) -C eval
	./eval/evaluate $(WHICH) data/smoke.bin models/$(MODEL)/ckpts/*.nd 2>/dev/null || \
	./eval/evaluate $(WHICH) data/smoke.bin models/01-sanity/ckpts/sanity.nd

infer: framework
	clang -std=c11 -O2 -Iframework/include inference/generate.c -Lframework -lneedle -lm -o inference/generate
	./inference/generate models/01-sanity/ckpts/sanity.nd

# User-facing CLI shape (Cactus parity). Exit 2 until text→tool path exists.
chat-demo:
	clang -std=c11 -O2 inference/chat_demo.c -o inference/chat_demo
	./inference/chat_demo --query "What's the weather in San Francisco?" \
	  --tools '[{"name":"get_weather","parameters":{"location":"string"}}]' || true

# Web UI on 0.0.0.0:7860 (Cactus-style port). Ctrl+C to stop.
# Text→tool NOT live yet — form + stub/reference only.
PORT ?= 7860
playground:
	clang -std=c11 -O2 inference/playground.c -o inference/playground
	@echo "→ open http://127.0.0.1:$(PORT)/  (binds 0.0.0.0)"
	./inference/playground $(PORT)

clean:
	$(MAKE) -C framework clean
	$(MAKE) -C tests clean
	-$(MAKE) -C models/01-sanity clean
	-$(MAKE) -C models/02-pilot-6m clean
	-$(MAKE) -C models/03-full-26m clean
	-$(MAKE) -C eval clean
	rm -f inference/generate inference/chat_demo inference/playground
