import type { ImgHTMLAttributes } from 'react';

type IconProps = ImgHTMLAttributes<HTMLImageElement>;

const ICON_SIZE = 'size-4';

/**
 * Social link icons.
 *
 * Coloured SVG icons tinted to the CppTorch navy `#14213D`, served as ``
 * tags pointing to publicly available SVG sources (Simple Icons CDN).
 */
export function GitHubIcon(props: IconProps) {
  return (
    <img
      aria-hidden="true"
      src="https://cdn.simpleicons.org/github/14213D"
      alt=""
      width={24}
      height={24}
      className={ICON_SIZE}
      {...props}
    />
  );
}

export function LinkedInIcon(props: IconProps) {
  return (
    <img
      aria-hidden="true"
      src="data:image/svg+xml,%3Csvg%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22%2314213D%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%3E%3Cpath%20d%3D%22M20.447%2020.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853%200-2.136%201.445-2.136%202.939v5.667H9.351V9h3.414v1.561h.046c.477-.9%201.637-1.85%203.37-1.85%203.601%200%204.267%202.37%204.267%205.455v6.286zM5.337%207.433a2.062%202.062%200%2001-2.063-2.065%202.064%202.064%200%20112.063%202.065zm1.782%2013.019H3.555V9h3.564v11.452zM22.225%200H1.771C.792%200%200%20.774%200%201.729v20.542C0%2023.227.792%2024%201.771%2024h20.451C24%2023.227%2024%2022.271%2024%2021.271V1.729C24%20.774%2023.2%200%2022.222%200z%22%2F%3E%3C%2Fsvg%3E"
      alt=""
      width={24}
      height={24}
      className={ICON_SIZE}
      {...props}
    />
  );
}

export function YouTubeIcon(props: IconProps) {
  return (
    <img
      aria-hidden="true"
      src="https://cdn.simpleicons.org/youtube/14213D"
      alt=""
      width={24}
      height={24}
      className={ICON_SIZE}
      {...props}
    />
  );
}
