import { highlight } from 'fumadocs-core/highlight';
import { CodeBlock } from 'fumadocs-ui/components/codeblock';

type SupportedLanguage = 'cpp' | 'bash' | 'cmake' | 'json' | 'text' | 'yaml';

export interface CodeSampleProps {
  code: string;
  lang?: SupportedLanguage;
  title?: string;
}

/**
 * Renders a server-side highlighted code block, used outside of MDX pages.
 */
export async function CodeSample({ code, lang = 'cpp', title }: CodeSampleProps) {
  const highlighted = await highlight(code, { lang: lang as 'cpp' });

  return (
    <CodeBlock title={title} className="my-0">
      {highlighted}
    </CodeBlock>
  );
}
