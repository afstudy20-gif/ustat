#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
BUILD_ROOT="$ROOT_DIR/build/macos"
RELEASE_DIR="$ROOT_DIR/release"
APP_NAME="uSTAT"
PKG_IDENTIFIER="uk.drtr.ustat"
VERSION="$(node -p "require('$FRONTEND_DIR/package.json').version")"
ARCH="$(uname -m)"

export COPYFILE_DISABLE=1

mkdir -p "$BUILD_ROOT" "$RELEASE_DIR"

echo "Building frontend..."
npm --prefix "$FRONTEND_DIR" ci --legacy-peer-deps
npm --prefix "$FRONTEND_DIR" run build

echo "Preparing Python build environment..."
rm -rf "$BUILD_ROOT/venv"
python3 -m venv "$BUILD_ROOT/venv"
"$BUILD_ROOT/venv/bin/python" -m pip install --upgrade pip wheel setuptools
"$BUILD_ROOT/venv/bin/python" -m pip install -r "$ROOT_DIR/backend/requirements.txt" pyinstaller

ICON_ARGS=()
ICONSET="$BUILD_ROOT/uSTAT.iconset"
ICNS="$BUILD_ROOT/uSTAT.icns"
if command -v sips >/dev/null 2>&1 && command -v iconutil >/dev/null 2>&1; then
  rm -rf "$ICONSET"
  mkdir -p "$ICONSET"
  for size in 16 32 64 128 256 512; do
    sips -z "$size" "$size" "$FRONTEND_DIR/public/pwa-512.png" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
  done
  for size in 16 32 128 256; do
    scale=$((size * 2))
    cp "$ICONSET/icon_${scale}x${scale}.png" "$ICONSET/icon_${size}x${size}@2x.png"
  done
  cp "$ICONSET/icon_512x512.png" "$ICONSET/icon_512x512@2x.png"
  iconutil -c icns "$ICONSET" -o "$ICNS"
  ICON_ARGS=(--icon "$ICNS")
fi

echo "Bundling macOS app..."
rm -rf "$BUILD_ROOT/pyinstaller-build" "$BUILD_ROOT/pyinstaller-dist"
"$BUILD_ROOT/venv/bin/pyinstaller" \
  --noconfirm \
  --windowed \
  --name "$APP_NAME" \
  --paths "$ROOT_DIR/backend" \
  --add-data "$ROOT_DIR/backend:backend" \
  --add-data "$FRONTEND_DIR/dist:frontend/dist" \
  --collect-all pyreadstat \
  --collect-submodules scipy \
  --collect-submodules sklearn \
  --collect-submodules statsmodels \
  --collect-submodules lifelines \
  --collect-submodules patsy \
  "${ICON_ARGS[@]}" \
  --specpath "$BUILD_ROOT" \
  --distpath "$BUILD_ROOT/pyinstaller-dist" \
  --workpath "$BUILD_ROOT/pyinstaller-build" \
  "$ROOT_DIR/macos/ustat_launcher.py"

APP_PATH="$BUILD_ROOT/pyinstaller-dist/$APP_NAME.app"
PKG_PATH="$RELEASE_DIR/${APP_NAME}-${VERSION}-macos-${ARCH}.pkg"
PKG_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ustat-pkg-root.XXXXXX")"
PKG_SCRIPTS="$(mktemp -d "${TMPDIR:-/tmp}/ustat-pkg-scripts.XXXXXX")"

find "$APP_PATH" \( -name '._*' -o -name '.DS_Store' \) -delete
xattr -cr "$APP_PATH" 2>/dev/null || true
mkdir -p "$PKG_ROOT/Applications"
ditto --norsrc --noextattr "$APP_PATH" "$PKG_ROOT/Applications/$APP_NAME.app"
xattr -cr "$PKG_ROOT" 2>/dev/null || true

COPYFILE_DISABLE=1 tar -czf "$PKG_SCRIPTS/$APP_NAME.app.tar.gz" -C "$PKG_ROOT/Applications" "$APP_NAME.app"
cat > "$PKG_SCRIPTS/postinstall" <<EOF
#!/bin/sh
set -e

SCRIPT_DIR="\$(cd "\$(dirname "\$0")" && pwd)"
APP_NAME="$APP_NAME"
INSTALL_DIR="\${INSTALL_DIR:-/Applications}"

mkdir -p "\$INSTALL_DIR"
rm -rf "\$INSTALL_DIR/\$APP_NAME.app"
tar -xzf "\$SCRIPT_DIR/\$APP_NAME.app.tar.gz" -C "\$INSTALL_DIR"
chown -R root:wheel "\$INSTALL_DIR/\$APP_NAME.app" 2>/dev/null || true
chmod -R go+rX "\$INSTALL_DIR/\$APP_NAME.app"
xattr -cr "\$INSTALL_DIR/\$APP_NAME.app" 2>/dev/null || true

exit 0
EOF
chmod 755 "$PKG_SCRIPTS/postinstall"

echo "Creating installer package..."
rm -f "$PKG_PATH"
pkgbuild \
  --nopayload \
  --scripts "$PKG_SCRIPTS" \
  --identifier "$PKG_IDENTIFIER" \
  --version "$VERSION" \
  "$PKG_PATH"

echo "Created $PKG_PATH"
