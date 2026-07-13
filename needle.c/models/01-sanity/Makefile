CC = clang
CFLAGS = -std=c11 -O2 -Wall -Wextra -I../../framework/include
LDFLAGS = -L../../framework -lneedle -lm
ROOT = ../..

.PHONY: smoke train fc data clean

../../framework/libneedle.a:
	$(MAKE) -C ../../framework

data:
	$(CC) $(CFLAGS) $(ROOT)/tools/write_smoke_bin.c $(LDFLAGS) -o $(ROOT)/tools/write_smoke_bin
	mkdir -p $(ROOT)/data
	$(ROOT)/tools/write_smoke_bin $(ROOT)/data/smoke.bin 64
	$(ROOT)/tools/write_smoke_bin $(ROOT)/data/small_train.bin 256

# fixture path (synthetic token ids, vocab=64)
train_bin: src/main.c ../../framework/libneedle.a
	$(CC) $(CFLAGS) src/main.c $(LDFLAGS) -o train_bin

# shared train entry (supports mode=fc + BPE vocab)
fc_bin: ../../training/train_main.c ../../framework/libneedle.a
	$(CC) $(CFLAGS) ../../training/train_main.c $(LDFLAGS) -o fc_bin

smoke: data train_bin
	mkdir -p ckpts
	./train_bin smoke ../../data/smoke.bin

train: data train_bin
	mkdir -p ckpts
	./train_bin train ../../data/small_train.bin

# real FC path: BPE vocab, data/fc_train.bin → ckpts/sanity_fc.nd
fc: fc_bin
	mkdir -p ckpts
	./fc_bin A fc ../../data/fc_train.bin ckpts/sanity_fc.nd results.tsv \
	  ../../tokenizer/vocab.json ../../tokenizer/merges.txt

clean:
	rm -f train_bin fc_bin
