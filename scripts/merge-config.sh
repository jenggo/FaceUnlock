#!/bin/bash
# Merge new config keys from default into user config, preserving user values.
# Usage: merge-config.sh <user_config> <default_config> <output>

USER_CFG="$1"
DEFAULT_CFG="$2"
OUTPUT="$3"

if [ ! -f "$USER_CFG" ] || [ ! -f "$DEFAULT_CFG" ]; then
    echo "Error: config files not found" >&2
    exit 1
fi

# Collect existing keys from user config (skip comments and section headers)
USER_KEYS=$(grep -E '^\s*[a-z_]+\s*=' "$USER_CFG" | sed 's/\s*=.*//' | sed 's/^\s*//')

# Find keys in default that are missing from user
MISSING=""
CURRENT_SECTION=""
while IFS= read -r line; do
    trimmed=$(echo "$line" | sed 's/^\s*//;s/\s*$//')
    
    # Track section
    if echo "$trimmed" | grep -qE '^\[.*\]$'; then
        CURRENT_SECTION="$trimmed"
        continue
    fi
    
    # Skip comments and empty lines
    if echo "$trimmed" | grep -qE '^(#|$)'; then
        continue
    fi
    
    # Extract key
    if echo "$trimmed" | grep -q '='; then
        KEY=$(echo "$trimmed" | sed 's/\s*=.*//' | sed 's/^\s*//')
        
        # Check if key exists in user config
        if ! echo "$USER_KEYS" | grep -qxF "$KEY"; then
            MISSING="${MISSING}${CURRENT_SECTION}|${line}"$'\n'
        fi
    fi
done < "$DEFAULT_CFG"

# If nothing missing, just copy user config
if [ -z "$(echo "$MISSING" | sed '/^$/d')" ]; then
    cp "$USER_CFG" "$OUTPUT"
    exit 0
fi

# Start with user config
cp "$USER_CFG" "$OUTPUT"

# Append missing keys grouped by section
PREV_SECTION=""
echo "$MISSING" | while IFS='|' read -r section line; do
    [ -z "$line" ] && continue
    
    # If section changed, check if it exists in output
    if [ "$section" != "$PREV_SECTION" ] && [ -n "$section" ]; then
        if ! grep -qF "$section" "$OUTPUT"; then
            # New section - append section header
            echo "" >> "$OUTPUT"
            echo "$section" >> "$OUTPUT"
        fi
        PREV_SECTION="$section"
    fi
    
    echo "$line" >> "$OUTPUT"
done
