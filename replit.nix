{pkgs}: {
  deps = [
    pkgs.xorg.libXrender
    pkgs.xorg.libXext
    pkgs.xorg.libX11
    pkgs.xorg.libxcb
  ];
}
