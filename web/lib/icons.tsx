import type { Icon } from '@phosphor-icons/react';
import {
  BrainIcon,
  CubeIcon,
  CpuIcon,
  DatabaseIcon,
  GaugeIcon,
  GitBranchIcon,
  GraduationCapIcon,
  GridFourIcon,
  HeartbeatIcon,
  LightningIcon,
  NetworkIcon,
  PackageIcon,
  PuzzlePieceIcon,
  RocketIcon,
  SigmaIcon,
  SparkleIcon,
  StackIcon,
  TargetIcon,
  UserIcon,
} from '@phosphor-icons/react/ssr';
import type { ReactNode } from 'react';

/**
 * The icon set of the documentation site.
 *
 * Phosphor's SSR build is used on purpose: the icons are rendered inside
 * server components and inside the page tree that is handed over to the
 * (client side) layouts.
 *
 * All icons below are rendered with `weight="thin"` so they appear outlined
 * instead of filled.
 */
export const icons = {
  Brain: BrainIcon,
  Cube: CubeIcon,
  Cpu: CpuIcon,
  Database: DatabaseIcon,
  Gauge: GaugeIcon,
  GitBranch: GitBranchIcon,
  GraduationCap: GraduationCapIcon,
  GridFour: GridFourIcon,
  Heartbeat: HeartbeatIcon,
  Lightning: LightningIcon,
  Network: NetworkIcon,
  Package: PackageIcon,
  PuzzlePiece: PuzzlePieceIcon,
  Rocket: RocketIcon,
  Sigma: SigmaIcon,
  Sparkle: SparkleIcon,
  Stack: StackIcon,
  Target: TargetIcon,
  User: UserIcon,
} satisfies Record<string, Icon>;

export type IconName = keyof typeof icons;

/**
 * Resolves the `icon` frontmatter of a page or folder into a Phosphor icon,
 * using the outlined (thin) weight.
 */
export function resolveIcon(name: string | undefined): ReactNode {
  if (!name) return undefined;

  const Icon = icons[name as IconName];
  if (!Icon) {
    console.warn(`[icons] Unknown icon detected: ${name}.`);
    return undefined;
  }

  return <Icon weight="thin" />;
}
