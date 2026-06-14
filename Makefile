.PHONY: all daemon enroll-cli pam install uninstall clean

all: daemon enroll-cli pam

daemon:
	cargo build --release --manifest-path daemon/Cargo.toml

enroll-cli:
	cargo build --release --manifest-path enroll-cli/Cargo.toml

pam:
	$(MAKE) -C pam

install:
	install -d /usr/bin
	install -m 755 target/release/faceunlockd /usr/bin/faceunlockd
	install -m 755 target/release/faceunlock /usr/bin/faceunlock
	$(MAKE) -C pam install
	install -d /etc/faceunlock
	@if [ -f /etc/faceunlock/faceunlock.toml ]; then \
		install -m 644 config/faceunlock.toml /etc/faceunlock/faceunlock.toml.default; \
		./scripts/merge-config.sh /etc/faceunlock/faceunlock.toml /etc/faceunlock/faceunlock.toml.default /etc/faceunlock/faceunlock.toml; \
	else \
		install -m 644 config/faceunlock.toml /etc/faceunlock/faceunlock.toml; \
	fi
	install -d /usr/share/faceunlock/models
	install -d /etc/systemd/system
	install -m 644 systemd/faceunlockd.service /etc/systemd/system/faceunlockd.service
	install -d /var/lib/faceunlock/embeddings
	-chown root:faceunlock /var/lib/faceunlock /var/lib/faceunlock/embeddings
	-chmod 770 /var/lib/faceunlock /var/lib/faceunlock/embeddings
	-groupadd -r faceunlock 2>/dev/null || true
	touch /var/log/faceunlock.log
	chmod 666 /var/log/faceunlock.log
	systemctl daemon-reload
	-systemctl restart faceunlockd

uninstall:
	-systemctl stop faceunlockd
	-systemctl disable faceunlockd
	rm -f /usr/bin/faceunlockd
	rm -f /usr/bin/faceunlock
	$(MAKE) -C pam uninstall
	rm -rf /etc/faceunlock
	rm -f /etc/systemd/system/faceunlockd.service
	rm -f /var/log/faceunlock.log
	systemctl daemon-reload

clean:
	cargo clean --manifest-path daemon/Cargo.toml
	cargo clean --manifest-path enroll-cli/Cargo.toml
	$(MAKE) -C pam clean
