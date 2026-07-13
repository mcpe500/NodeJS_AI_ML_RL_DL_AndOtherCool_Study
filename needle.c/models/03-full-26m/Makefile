CC = clang
CFLAGS = -std=c11 -O2 -Wall -Wextra -I../../framework/include
LDFLAGS = -L../../framework -lneedle -lm

.PHONY: smoke train data clean

../../framework/libneedle.a:
	$(MAKE) -C ../../framework

data:
	$(MAKE) -C ../01-sanity data

train_bin: ../../training/train_main.c ../../framework/libneedle.a
	$(CC) $(CFLAGS) ../../training/train_main.c $(LDFLAGS) -o train_bin

smoke: data train_bin
	mkdir -p ckpts
	./train_bin C smoke ../../data/smoke.bin ckpts/full.nd results.tsv

train: data train_bin
	mkdir -p ckpts
	./train_bin C train ../../data/small_train.bin ckpts/full.nd results.tsv

clean:
	rm -f train_bin
