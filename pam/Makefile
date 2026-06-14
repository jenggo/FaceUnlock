CC = gcc
CFLAGS = -Wall -Wextra -fPIC
LDFLAGS = -shared
LIBS = -lpam -ljson-c

TARGET = pam_faceunlock.so

all: $(TARGET)

$(TARGET): pam_faceunlock.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

install: $(TARGET)
	install -m 644 $(TARGET) /usr/lib/security/$(TARGET)

uninstall:
	rm -f /usr/lib/security/$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all install uninstall clean
