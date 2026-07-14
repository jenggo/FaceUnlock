#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/time.h>
#include <security/pam_modules.h>
#include <security/pam_ext.h>
#include <json-c/json.h>

#define SOCKET_PATH "/run/faceunlockd/auth.sock"
#define TIMEOUT_SECONDS 5
#define BUFFER_SIZE 4096

static int send_request(const char *action, const char *username, char *response, size_t response_len) {
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        return -1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(sock);
        return -1;
    }

    struct timeval tv;
    tv.tv_sec = TIMEOUT_SECONDS;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    json_object *req = json_object_new_object();
    json_object_object_add(req, "action", json_object_new_string(action));
    if (username) {
        json_object_object_add(req, "user", json_object_new_string(username));
    }

    const char *json_str = json_object_to_json_string(req);
    size_t json_len = strlen(json_str);

    char *buf = malloc(json_len + 2);
    if (!buf) {
        json_object_put(req);
        close(sock);
        return -1;
    }
    memcpy(buf, json_str, json_len);
    buf[json_len] = '\n';
    buf[json_len + 1] = '\0';

    json_object_put(req);

    ssize_t written = write(sock, buf, json_len + 1);
    free(buf);

    if (written < 0) {
        close(sock);
        return -1;
    }

    size_t total = 0;
    while (total < response_len - 1) {
        ssize_t n = read(sock, response + total, response_len - 1 - total);
        if (n <= 0) {
            break;
        }
        total += n;
        if (response[total - 1] == '\n') {
            break;
        }
    }
    response[total] = '\0';

    close(sock);
    return (int)total;
}

PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    (void)flags; (void)argc; (void)argv;
    const char *username = NULL;
    int retval = pam_get_user(pamh, &username, NULL);
    if (retval != PAM_SUCCESS || !username) {
        return PAM_AUTH_ERR;
    }

    char response[BUFFER_SIZE];
    int len = send_request("authenticate", username, response, sizeof(response));

    if (len < 0) {
        return PAM_AUTHINFO_UNAVAIL;
    }

    if (len == 0) {
        return PAM_AUTH_ERR;
    }

    json_object *resp = json_tokener_parse(response);
    if (!resp) {
        return PAM_AUTH_ERR;
    }

    json_object *result_obj;
    if (!json_object_object_get_ex(resp, "result", &result_obj)) {
        json_object_put(resp);
        return PAM_AUTH_ERR;
    }

    const char *result = json_object_get_string(result_obj);
    int auth_result = PAM_AUTH_ERR;

    if (strcmp(result, "ok") == 0) {
        auth_result = PAM_SUCCESS;
    } else {
        /* Check for structured reason_code */
        json_object *reason_code_obj;
        const char *reason_code = NULL;
        if (json_object_object_get_ex(resp, "reason_code", &reason_code_obj)) {
            reason_code = json_object_get_string(reason_code_obj);
        }

        const char *message = NULL;
        if (reason_code) {
            if (strcmp(reason_code, "camera_blocked") == 0) {
                message = "Face unlock failed: camera appears blocked. Open the lid or privacy shutter.";
            } else if (strcmp(reason_code, "no_face") == 0) {
                message = "Face unlock failed: no face detected.";
            } else if (strcmp(reason_code, "score_below_threshold") == 0) {
                message = "Face unlock failed: face not recognized.";
            }
        }

        if (message) {
            pam_error(pamh, "%s", message);
        } else {
            /* Fallback: use free-text reason for diagnostics */
            json_object *reason_obj;
            const char *reason = NULL;
            if (json_object_object_get_ex(resp, "reason", &reason_obj)) {
                reason = json_object_get_string(reason_obj);
            }
            pam_error(pamh, "Face unlock failed.");
            if (reason) {
                pam_syslog(pamh, LOG_ERR, "faceunlock reason: %s", reason);
            }
        }
    }

    json_object_put(resp);
    return auth_result;
}

PAM_EXTERN int pam_sm_setcred(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    (void)pamh; (void)flags; (void)argc; (void)argv;
    return PAM_SUCCESS;
}

PAM_EXTERN int pam_sm_acct_mgmt(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    (void)pamh; (void)flags; (void)argc; (void)argv;
    return PAM_SUCCESS;
}

PAM_EXTERN int pam_sm_open_session(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    (void)pamh; (void)flags; (void)argc; (void)argv;
    return PAM_SUCCESS;
}

PAM_EXTERN int pam_sm_close_session(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    (void)pamh; (void)flags; (void)argc; (void)argv;
    return PAM_SUCCESS;
}

PAM_EXTERN int pam_sm_chauthtok(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    (void)pamh; (void)flags; (void)argc; (void)argv;
    return PAM_SUCCESS;
}
