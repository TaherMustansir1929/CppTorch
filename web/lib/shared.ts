import { createGetUrl } from 'fumadocs-core/source';

export const appName = 'CppTorch';
export const docsRoute = '/docs';
export const docsImageRoute = '/og/docs';

// fill this with your actual GitHub info, for example:
export const gitConfig = {
  user: 'TaherMustansir1929',
  repo: 'CppTorch',
  branch: 'main',
};

/**
 * Social profiles of the project author, shown in the navbar and the footer.
 */
export const socialLinks = {
  github: 'https://github.com/TaherMustansir1929',
  linkedin: 'https://www.linkedin.com/in/taher-mustansir-aa6340381/',
  youtube: 'https://www.youtube.com/@ThePitza',
} as const;

export const repoUrl = `https://github.com/${gitConfig.user}/${gitConfig.repo}`;

const getImageUrl = createGetUrl(docsImageRoute);

export function getPageImageUrl(page: { slugs: string[]; locale?: string }) {
  const segments = [...page.slugs, 'image.png'];

  return { segments, url: getImageUrl(segments, page.locale) };
}
