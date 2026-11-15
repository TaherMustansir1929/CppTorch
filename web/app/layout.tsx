import { RootProvider } from 'fumadocs-ui/provider/next';
import DefaultSearchDialog from 'fumadocs-ui/components/dialog/search-default';
import './global.css';
import { Roboto, Source_Serif_4 } from 'next/font/google';
import type { Metadata } from 'next';
import { appName, repoUrl, socialLinks } from '@/lib/shared';

const roboto = Roboto({
  subsets: ['latin'],
  variable: '--font-roboto',
  display: 'swap',
});

const serif = Source_Serif_4({
  subsets: ['latin'],
  variable: '--font-heading',
  display: 'swap',
});

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? 'http://localhost:3000'),
  title: {
    default: `${appName} - a minimal neural networks library in C++20`,
    template: `%s | ${appName}`,
  },
  description:
    'CppTorch is a minimal neural networks library in C++20 with numpy-style array primitives, dense layers, activations, losses, optimizers and a complete training loop.',
  keywords: [
    'CppTorch',
    'C++20',
    'neural network',
    'deep learning',
    'machine learning',
    'header-only',
    'BLAS',
    'OpenMP',
  ],
  authors: [{ name: 'Taher Mustansir', url: socialLinks.github }],
  openGraph: {
    type: 'website',
    siteName: appName,
    title: `${appName} — a minimal neural networks library in C++20`,
    description:
      'numpy-style arrays, dense layers, activations, losses, optimizers and a complete training loop — in C++20.',
  },
  twitter: {
    card: 'summary_large_image',
    title: appName,
    description: 'A minimal neural networks library in C++20.',
  },
  alternates: {
    canonical: repoUrl,
  },
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html
      lang="en"
      className={`${roboto.variable} ${serif.variable} light`}
      suppressHydrationWarning
    >
      <body className="flex flex-col min-h-screen">
        <RootProvider
          theme={{
            enabled: true,
            defaultTheme: 'light',
            forcedTheme: 'light',
            attribute: 'class',
            disableTransitionOnChange: true,
          }}
          search={{
            SearchDialog: DefaultSearchDialog,
            options: { api: '/api/search' },
          }}
        >
          {children}
        </RootProvider>
      </body>
    </html>
  );
}
