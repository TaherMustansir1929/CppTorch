import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { GitHubIcon, LinkedInIcon, YouTubeIcon } from '@/components/social-icons';
import { appName, socialLinks } from './shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      // JSX supported
      title: (
        <span className="inline-flex items-center gap-2 font-bold tracking-tight">
          <img
            src="/cpptorch.png"
            alt="CppTorch"
            width="32"
            height="32"
            className="rounded-none"
          />
          {appName}
        </span>
      ),
    },
    // Fumadocs renders a GitHub icon for `githubUrl` on its own.
    githubUrl: socialLinks.github,
    // Disable the theme switcher — the site stays in light mode only.
    themeSwitch: { enabled: false },
    // Enable search triggers in the navbar
    searchToggle: { enabled: true },
    links: [
      {
        type: 'icon',
        label: 'LinkedIn',
        text: 'LinkedIn',
        url: socialLinks.linkedin,
        icon: <LinkedInIcon className="size-4" />,
        external: true,
      },
      {
        type: 'icon',
        label: 'YouTube',
        text: 'YouTube',
        url: socialLinks.youtube,
        icon: <YouTubeIcon className="size-4" />,
        external: true,
      },
    ],
  };
}
