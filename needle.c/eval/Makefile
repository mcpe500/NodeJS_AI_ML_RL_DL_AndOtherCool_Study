CC = clang
CFLAGS = -std=c11 -O2 -Wall -Wextra -I../framework/include
LDFLAGS = -L../framework -lneedle -lm

.PHONY: all clean

all: evaluate

evaluate: evaluate.c ../framework/libneedle.a
	$(CC) $(CFLAGS) evaluate.c $(LDFLAGS) -o evaluate

clean:
	rm -f evaluate
