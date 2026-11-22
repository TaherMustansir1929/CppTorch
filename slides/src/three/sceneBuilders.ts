import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import gsap from 'gsap';
import type { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  applyNodeInflationClick,
  BENCHMARK_ITEMS,
  decayNodeInflation,
  evaluateLeakyRelu,
  generateOptimizerTrajectory,
  getMatrixHighlightIndices,
  sampleNavierStokesVorticity,
  type OptimizerName,
} from '../utils/deckMath';

export const PALETTE = {
  cream: 0xf4ede0,
  creamHex: '#f4ede0',
  navy: 0x14213d,
  navyHex: '#14213d',
  navySoft: 0x223661,
  coral: 0xff6b5b,
  coralHex: '#ff6b5b',
  white: 0xffffff,
};

export interface SlideSceneState {
  pulseTrigger: number;
  selectedFeatureIndex: number | null;
  selectedPillarIndex: number | null;
  matrixSize: number;
  selectedCell: [number, number];
  benchmarkMode: 'epoch' | 'operation';
  replayTrigger: number;
  networkDepth: 'shallow' | 'medium' | 'deep';
  selectedNeuronId: number | null;
  reluAlpha: number;
  mseSampleIndex: number;
  selectedOptimizer: OptimizerName;
  loopStage: 'all' | 'Forward' | 'Backward' | 'Update';
  loopSpeed: number;
  datasetType: 'spiral' | 'housing' | 'audio' | 'svm';
  scalerFitMode: boolean;
  resampleTrigger: number;
  exampleFocus: 'both' | 'svm' | 'navier';
  snapSvmPlaneTrigger: number;
  selectedStep: number | null;
  burstTrigger: number;
  selectedAuthorIndex: number | null;
}

export interface BuiltScene {
  group: THREE.Group;
  cameraPos: THREE.Vector3;
  cameraTarget: THREE.Vector3;
  enableOrbit: boolean;
  interactables: THREE.Object3D[];
  update: (elapsed: number, delta: number, state: SlideSceneState) => void;
  onStateChange: (
    state: SlideSceneState,
    camera: THREE.PerspectiveCamera,
    controls: OrbitControls | null
  ) => void;
  onSelect: (hit: THREE.Intersection) => void;
  dispose: () => void;
}

export interface SceneCallbacks {
  onSelectFeature?: (index: number) => void;
  onSelectPillar?: (index: number) => void;
  onSelectMatrixCell?: (row: number, col: number) => void;
  onSelectNeuron?: (id: number | null) => void;
  onSelectOptimizer?: (opt: OptimizerName) => void;
  onSelectLoopStage?: (stage: 'all' | 'Forward' | 'Backward' | 'Update') => void;
  onSelectStep?: (step: number | null) => void;
  onSelectAuthor?: (idx: number | null) => void;
  onTriggerPulse?: () => void;
  onSnapSvm?: () => void;
  onToggleFitMode?: () => void;
  onResampleData?: () => void;
  onSelectMseSample?: (idx: number) => void;
}

// Reusable scratch math objects (zero GC in loop)
const _dummy = new THREE.Object3D();
const _vecA = new THREE.Vector3();
const _vecB = new THREE.Vector3();
const _color = new THREE.Color();

function createContactShadow(radius = 3.2): THREE.Mesh {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    const grad = ctx.createRadialGradient(64, 64, 4, 64, 64, 62);
    grad.addColorStop(0, 'rgba(20, 33, 61, 0.24)');
    grad.addColorStop(0.55, 'rgba(20, 33, 61, 0.08)');
    grad.addColorStop(1, 'rgba(20, 33, 61, 0.0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 128, 128);
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const geo = new THREE.CircleGeometry(radius, 32);
  geo.rotateX(-Math.PI / 2);
  const mat = new THREE.MeshBasicMaterial({
    map: tex,
    transparent: true,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.y = -2.1;
  return mesh;
}

interface LabelCardOptions {
  title: string;
  subtitle?: string;
  badge?: string;
  compact?: boolean;
  onClick?: () => void;
}

interface LabelCardRefs {
  obj: CSS2DObject;
  el: HTMLElement;
  badgeEl: HTMLElement | null;
  titleEl: HTMLElement;
  subEl: HTMLElement | null;
}

function create3DLabel({
  title,
  subtitle,
  badge,
  compact = false,
  onClick,
}: LabelCardOptions): LabelCardRefs {
  const el = document.createElement(onClick ? 'button' : 'div');
  if (el instanceof HTMLButtonElement) {
    el.type = 'button';
  }
  el.style.pointerEvents = onClick ? 'auto' : 'none';
  el.style.cursor = onClick ? 'pointer' : 'default';
  el.style.userSelect = 'none';
  el.style.backgroundColor = '#14213d';
  el.style.color = '#ffffff';
  el.style.border = '2px solid #ff6b5b';
  el.style.borderRadius = '0px';
  el.style.padding = compact ? '5px 10px' : '8px 14px';
  el.style.textAlign = 'center';
  el.style.fontFamily = '"Geist Mono", ui-monospace, monospace';
  el.style.lineHeight = '1.25';
  el.style.whiteSpace = 'nowrap';
  el.style.boxShadow = onClick
    ? '0 4px 0 rgba(20, 33, 61, 0.34), 0 6px 14px rgba(20, 33, 61, 0.22)'
    : '0 4px 12px rgba(20, 33, 61, 0.22)';
  el.style.transition =
    'opacity 200ms ease, border-color 200ms ease, background-color 200ms ease, color 200ms ease';

  let badgeEl: HTMLElement | null = null;
  if (badge) {
    badgeEl = document.createElement('div');
    badgeEl.textContent = badge;
    badgeEl.style.fontSize = '10px';
    badgeEl.style.fontWeight = '700';
    badgeEl.style.color = '#ff6b5b';
    badgeEl.style.marginBottom = '2px';
    badgeEl.style.letterSpacing = '0.04em';
    el.appendChild(badgeEl);
  }

  const titleEl = document.createElement('div');
  titleEl.textContent = title;
  titleEl.style.fontSize = compact ? '12px' : '14px';
  titleEl.style.fontWeight = '700';
  titleEl.style.color = '#ffffff';
  el.appendChild(titleEl);

  let subEl: HTMLElement | null = null;
  if (subtitle) {
    subEl = document.createElement('div');
    subEl.textContent = subtitle;
    subEl.style.fontFamily = '"Roboto", system-ui, sans-serif';
    subEl.style.fontSize = compact ? '10px' : '11px';
    subEl.style.fontWeight = '500';
    subEl.style.color = '#f4ede0';
    subEl.style.marginTop = '2px';
    el.appendChild(subEl);
  }

  if (onClick) {
    el.addEventListener('click', (ev) => {
      ev.stopPropagation();
      onClick();
    });
  }

  const obj = new CSS2DObject(el);
  return { obj, el, badgeEl, titleEl, subEl };
}

function set3DLabelActive(card: LabelCardRefs, active: boolean) {
  card.el.style.backgroundColor = active ? '#ff6b5b' : '#14213d';
  card.el.style.borderColor = active ? '#14213d' : '#ff6b5b';
  card.el.style.color = active ? '#14213d' : '#ffffff';
  card.titleEl.style.color = active ? '#14213d' : '#ffffff';
  if (card.badgeEl) {
    card.badgeEl.style.color = active ? '#14213d' : '#ff6b5b';
  }
  if (card.subEl) {
    card.subEl.style.color = active ? '#14213d' : '#f4ede0';
  }
}

function addEdges(mesh: THREE.Mesh, color = PALETTE.coral, opacity = 0.85): THREE.LineSegments {
  const edges = new THREE.EdgesGeometry(mesh.geometry);
  const mat = new THREE.LineBasicMaterial({
    color,
    transparent: opacity < 1,
    opacity,
  });
  const seg = new THREE.LineSegments(edges, mat);
  mesh.add(seg);
  return seg;
}

function disposeObjectTree(root: THREE.Object3D) {
  root.traverse((obj) => {
    if (obj instanceof CSS2DObject) {
      obj.element.remove();
    }
    const mesh = obj as THREE.Mesh;
    if (mesh.geometry) {
      mesh.geometry.dispose();
    }
    const mat = mesh.material;
    if (Array.isArray(mat)) {
      mat.forEach((m) => {
        const std = m as THREE.MeshStandardMaterial;
        std.map?.dispose();
        m.dispose();
      });
    } else if (mat) {
      const std = mat as THREE.MeshStandardMaterial;
      std.map?.dispose();
      mat.dispose();
    }
  });
}

// ============================================================================
// SLIDE 1: Welcome Neural Node Cluster
// ============================================================================
function buildSlide1(callbacks: SceneCallbacks): BuiltScene {
  const group = new THREE.Group();
  group.add(createContactShadow(3.0));

  const cluster = new THREE.Group();
  group.add(cluster);

  // Outer architectural geometric frame (composite 3-part frame + cluster)
  const ringGeo = new THREE.BoxGeometry(4.2, 4.2, 0.08);
  const frameEdges = new THREE.EdgesGeometry(ringGeo);
  ringGeo.dispose();
  const frameWire = new THREE.LineSegments(
    frameEdges,
    new THREE.LineBasicMaterial({ color: PALETTE.navy, transparent: true, opacity: 0.25 })
  );
  frameWire.rotation.z = Math.PI / 4;
  group.add(frameWire);

  // Node positions in low-poly icosahedral shell + inner core
  const nodePositions: THREE.Vector3[] = [new THREE.Vector3(0, 0, 0)];
  const count = 18;
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const y = 1 - (i / (count - 1)) * 2;
    const radiusAtY = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = goldenAngle * i;
    const r = 1.65;
    nodePositions.push(
      new THREE.Vector3(Math.cos(theta) * radiusAtY * r, y * r, Math.sin(theta) * radiusAtY * r)
    );
  }

  const sphereGeo = new THREE.IcosahedronGeometry(0.19, 1);
  const nodeMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.34,
    metalness: 0.25,
  });
  const instanced = new THREE.InstancedMesh(sphereGeo, nodeMat, nodePositions.length);
  instanced.frustumCulled = false;

  const scales = new Float32Array(nodePositions.length).fill(1);
  nodePositions.forEach((pos, idx) => {
    _dummy.position.copy(pos);
    const s = idx === 0 ? 1.45 : 0.85 + (idx % 3) * 0.14;
    scales[idx] = s;
    _dummy.scale.setScalar(s);
    _dummy.updateMatrix();
    instanced.setMatrixAt(idx, _dummy.matrix);
    _color.setHex(idx === 0 || idx % 5 === 0 ? PALETTE.coral : PALETTE.navy);
    instanced.setColorAt(idx, _color);
  });
  instanced.instanceMatrix.needsUpdate = true;
  if (instanced.instanceColor) instanced.instanceColor.needsUpdate = true;
  instanced.computeBoundingSphere();
  cluster.add(instanced);

  // Coral & Navy connecting edges
  const linePts: number[] = [];
  for (let i = 0; i < nodePositions.length; i++) {
    for (let j = i + 1; j < nodePositions.length; j++) {
      if (nodePositions[i].distanceTo(nodePositions[j]) < 1.85) {
        linePts.push(
          nodePositions[i].x,
          nodePositions[i].y,
          nodePositions[i].z,
          nodePositions[j].x,
          nodePositions[j].y,
          nodePositions[j].z
        );
      }
    }
  }
  const edgeGeo = new THREE.BufferGeometry();
  edgeGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePts, 3));
  const edgeMat = new THREE.LineBasicMaterial({
    color: PALETTE.coral,
    transparent: true,
    opacity: 0.7,
  });
  const edgeLines = new THREE.LineSegments(edgeGeo, edgeMat);
  cluster.add(edgeLines);

  // Invisible click proxy sphere for responsive raycast
  const hitProxy = new THREE.Mesh(
    new THREE.SphereGeometry(2.1, 16, 16),
    new THREE.MeshBasicMaterial({ visible: false })
  );
  hitProxy.userData.kind = 'cluster';
  cluster.add(hitProxy);

  const ctx = gsap.context(() => {
    gsap.to(cluster.rotation, {
      y: Math.PI * 2,
      duration: 12,
      repeat: -1,
      ease: 'none',
    });
  });

  let lastPulse = -1;
  let targetInflation = 0;
  let displayInflation = 0;

  const triggerInflate = () => {
    targetInflation = applyNodeInflationClick(targetInflation);
  };

  return {
    group,
    cameraPos: new THREE.Vector3(0, 0.4, 6.2),
    cameraTarget: new THREE.Vector3(0, 0, 0),
    enableOrbit: false,
    interactables: [hitProxy],
    update: (elapsed, delta) => {
      frameWire.rotation.z = Math.PI / 4 + elapsed * 0.08;
      targetInflation = decayNodeInflation(targetInflation, delta);
      displayInflation += (targetInflation - displayInflation) * Math.min(1, delta * 14);
      cluster.scale.setScalar(1 + displayInflation * 0.14);

      nodePositions.forEach((pos, idx) => {
        const wave = Math.sin(elapsed * 2.4 + idx * 0.45) * 0.1;
        const s = scales[idx] * (1 + wave + displayInflation);
        _dummy.position.copy(pos);
        _dummy.scale.setScalar(s);
        _dummy.updateMatrix();
        instanced.setMatrixAt(idx, _dummy.matrix);
      });
      instanced.instanceMatrix.needsUpdate = true;
      edgeMat.opacity = 0.55 + Math.min(0.42, displayInflation * 0.55);
    },
    onStateChange: (state) => {
      if (lastPulse !== -1 && state.pulseTrigger !== lastPulse) {
        triggerInflate();
      }
      lastPulse = state.pulseTrigger;
    },
    onSelect: () => {
      triggerInflate();
      callbacks.onTriggerPulse?.();
    },
    dispose: () => {
      ctx.revert();
      disposeObjectTree(group);
    },
  };
}

// ============================================================================
// SLIDE 2: Four Floating Feature Cards
// ============================================================================
function buildSlide2(callbacks: SceneCallbacks): BuiltScene {
  const group = new THREE.Group();
  group.add(createContactShadow(4.2));

  const features = [
    { title: 'Header-Only', sub: 'include/cpptorch/' },
    { title: 'OpenMP+BLAS', sub: 'Parallel Kernels' },
    { title: 'Training Loop', sub: 'Fwd · Bwd · Update' },
    { title: 'Extensible', sub: 'Factory Registry' },
  ];

  const cardMeshes: THREE.Mesh[] = [];
  const cardGroups: THREE.Group[] = [];
  const borderMats: THREE.LineBasicMaterial[] = [];
  const baseMats: THREE.MeshStandardMaterial[] = [];
  const labelEls: HTMLElement[] = [];

  const cardGeo = new THREE.BoxGeometry(1.58, 1.95, 0.16);
  const accentStripGeo = new THREE.BoxGeometry(1.58, 0.12, 0.18);
  const pedestalGeo = new THREE.BoxGeometry(1.68, 0.12, 0.36);

  features.forEach((feat, i) => {
    const holder = new THREE.Group();
    const angle = (i - 1.5) * 0.28;
    holder.position.set((i - 1.5) * 2.34, 0.1, -Math.abs(i - 1.5) * 0.35);
    holder.rotation.y = -angle * 0.6;

    const mat = new THREE.MeshStandardMaterial({
      color: PALETTE.navy,
      roughness: 0.42,
      metalness: 0.22,
      transparent: true,
      opacity: 1,
    });
    const card = new THREE.Mesh(cardGeo.clone(), mat);
    card.userData.featureIndex = i;

    const border = addEdges(card, PALETTE.coral, 0.85);
    const strip = new THREE.Mesh(
      accentStripGeo.clone(),
      new THREE.MeshStandardMaterial({ color: PALETTE.coral, roughness: 0.25 })
    );
    strip.position.y = 0.92;
    strip.userData.featureIndex = i;

    const pedMat = new THREE.MeshStandardMaterial({
      color: PALETTE.navySoft,
      roughness: 0.65,
      metalness: 0.15,
      transparent: true,
      opacity: 1,
    });
    const pedestal = new THREE.Mesh(pedestalGeo.clone(), pedMat);
    pedestal.position.y = -1.18;
    addEdges(pedestal, PALETTE.navy, 0.65);

    const { obj: labelObj, el: labelEl } = create3DLabel({
      badge: `0${i + 1}`,
      title: feat.title,
      subtitle: feat.sub,
      compact: true,
      onClick: () => callbacks.onSelectFeature?.(i),
    });
    labelObj.position.set(0, 0.05, 0.14);
    holder.add(card, strip, pedestal, labelObj);
    group.add(holder);

    cardMeshes.push(card, strip);
    cardGroups.push(holder);
    borderMats.push(border.material as THREE.LineBasicMaterial);
    baseMats.push(mat);
    labelEls.push(labelEl);
  });

  cardGeo.dispose();
  accentStripGeo.dispose();
  pedestalGeo.dispose();

  const ctx = gsap.context(() => {
    cardGroups.forEach((cg, idx) => {
      gsap.fromTo(
        cg.position,
        { y: -0.15 },
        {
          y: 0.22,
          duration: 2.8,
          delay: idx * 0.35,
          yoyo: true,
          repeat: -1,
          ease: 'sine.inOut',
        }
      );
    });
  });

  return {
    group,
    cameraPos: new THREE.Vector3(0, 1.25, 10.6),
    cameraTarget: new THREE.Vector3(0, 0, 0),
    enableOrbit: false,
    interactables: cardMeshes,
    update: (elapsed) => {
      group.rotation.y = Math.sin(elapsed * 0.35) * 0.08;
    },
    onStateChange: (state) => {
      const sel = state.selectedFeatureIndex;
      cardGroups.forEach((cg, idx) => {
        const isActive = sel === null || sel === idx;
        const isPicked = sel === idx;
        gsap.to(cg.scale, {
          x: isPicked ? 1.18 : isActive ? 1 : 0.92,
          y: isPicked ? 1.18 : isActive ? 1 : 0.92,
          z: isPicked ? 1.18 : isActive ? 1 : 0.92,
          duration: 0.35,
          ease: 'power2.out',
        });
        baseMats[idx].opacity = isActive ? 1 : 0.38;
        borderMats[idx].color.setHex(isPicked ? PALETTE.coral : PALETTE.navySoft);
        labelEls[idx].style.opacity = isActive ? '1' : '0.4';
        labelEls[idx].style.borderColor = isPicked ? '#ff6b5b' : 'rgba(255, 107, 91, 0.55)';
      });
    },
    onSelect: (hit) => {
      const idx = hit.object.userData.featureIndex as number;
      if (typeof idx === 'number') {
        callbacks.onSelectFeature?.(idx);
      }
    },
    dispose: () => {
      ctx.revert();
      disposeObjectTree(group);
    },
  };
}

// ============================================================================
// SLIDE 3: Three Design Philosophy Pillars
// ============================================================================
function buildSlide3(callbacks: SceneCallbacks): BuiltScene {
  const group = new THREE.Group();
  group.add(createContactShadow(4.2));

  const labels = ['Header-Only', 'Extension Points', 'Minimal Core'];
  const pillarGroups: THREE.Group[] = [];
  const capMats: THREE.MeshStandardMaterial[] = [];
  const hitMeshes: THREE.Mesh[] = [];
  const labelCards: LabelCardRefs[] = [];

  labels.forEach((label, i) => {
    const pg = new THREE.Group();
    const x = (i - 1) * 2.45;
    pg.position.set(x, -1.5, 0);

    // 3-part composite pillar: plinth + shaft + glowing coral cap
    const plinth = new THREE.Mesh(
      new THREE.BoxGeometry(1.5, 0.25, 1.5),
      new THREE.MeshStandardMaterial({ color: PALETTE.navySoft, roughness: 0.65 })
    );
    plinth.position.y = 0.125;
    addEdges(plinth, PALETTE.navy, 0.9);

    const shaft = new THREE.Mesh(
      new THREE.BoxGeometry(1.18, 2.7, 1.18),
      new THREE.MeshStandardMaterial({ color: PALETTE.navy, roughness: 0.42, metalness: 0.22 })
    );
    shaft.position.y = 1.6;
    shaft.userData.pillarIndex = i;
    addEdges(shaft, PALETTE.coral, 0.75);

    const capMat = new THREE.MeshStandardMaterial({
      color: PALETTE.coral,
      emissive: PALETTE.coral,
      emissiveIntensity: 0.25,
      roughness: 0.22,
      metalness: 0.2,
    });
    const cap = new THREE.Mesh(new THREE.BoxGeometry(1.3, 0.22, 1.3), capMat);
    cap.position.y = 3.06;
    cap.userData.pillarIndex = i;
    addEdges(cap, PALETTE.navy, 0.9);

    const labelCard = create3DLabel({
      badge: `Rule 0${i + 1}`,
      title: label,
      compact: true,
      onClick: () => callbacks.onSelectPillar?.(i),
    });
    labelCard.obj.position.set(0, 1.75, 0.7);

    pg.add(plinth, shaft, cap, labelCard.obj);
    group.add(pg);

    pillarGroups.push(pg);
    capMats.push(capMat);
    hitMeshes.push(shaft, cap);
    labelCards.push(labelCard);
  });

  const ctx = gsap.context(() => {
    gsap.from(
      pillarGroups.map((p) => p.scale),
      {
        y: 0.05,
        duration: 0.75,
        stagger: 0.18,
        ease: 'back.out(1.7)',
      }
    );
    capMats.forEach((m, i) => {
      gsap.to(m, {
        emissiveIntensity: 0.65,
        duration: 1.8,
        delay: i * 0.25,
        yoyo: true,
        repeat: -1,
        ease: 'sine.inOut',
      });
    });
  });

  return {
    group,
    cameraPos: new THREE.Vector3(0, 1.6, 8.6),
    cameraTarget: new THREE.Vector3(0, 0.1, 0),
    enableOrbit: false,
    interactables: hitMeshes,
    update: (elapsed) => {
      group.rotation.y = Math.sin(elapsed * 0.4) * 0.12;
    },
    onStateChange: (state, camera, controls) => {
      const sel = state.selectedPillarIndex;
      pillarGroups.forEach((pg, idx) => {
        const active = sel === idx;
        gsap.to(pg.scale, {
          x: active ? 1.14 : 1,
          z: active ? 1.14 : 1,
          y: active ? 1.08 : 1,
          duration: 0.35,
          ease: 'power2.out',
        });
        set3DLabelActive(labelCards[idx], active);
      });
      const targetX = sel === null ? 0 : (sel - 1) * 2.25;
      gsap.to(camera.position, {
        x: targetX,
        y: sel === null ? 1.6 : 1.35,
        z: sel === null ? 8.6 : 6.9,
        duration: 0.65,
        ease: 'power2.out',
      });
      if (controls) {
        gsap.to(controls.target, {
          x: targetX,
          y: 0.1,
          z: 0,
          duration: 0.65,
          ease: 'power2.out',
          onUpdate: () => {
            controls.update();
          },
        });
      }
    },
    onSelect: (hit) => {
      const idx = hit.object.userData.pillarIndex as number;
      if (typeof idx === 'number') {
        callbacks.onSelectPillar?.(idx);
      }
    },
    dispose: () => {
      ctx.revert();
      disposeObjectTree(group);
    },
  };
}

// ============================================================================
// SLIDE 4: Matrix & NumCpp (C = A * B)
// ============================================================================
function buildSlide4(callbacks: SceneCallbacks): BuiltScene {
  const group = new THREE.Group();
  group.add(createContactShadow(4.5));

  const gridContainer = new THREE.Group();
  group.add(gridContainer);

  let currentSize = -1;
  let currentSelected: [number, number] = [0, 0];
  let cellMeshesA: THREE.Mesh[] = [];
  let cellMeshesB: THREE.Mesh[] = [];
  let cellMeshesC: THREE.Mesh[] = [];
  let beamLines: THREE.LineSegments | null = null;
  let interactables: THREE.Object3D[] = [];

  const cellGeo = new THREE.BoxGeometry(0.42, 0.42, 0.22);

  const rebuildGrid = (size: number, selectedCell: [number, number]) => {
    while (gridContainer.children.length > 0) {
      const child = gridContainer.children[0];
      gridContainer.remove(child);
      disposeObjectTree(child);
    }
    cellMeshesA = [];
    cellMeshesB = [];
    cellMeshesC = [];
    interactables = [];

    const spacing = 0.52;
    const offset = ((size - 1) * spacing) / 2;
    const { rowA, colB, cellC } = getMatrixHighlightIndices(
      selectedCell[0],
      selectedCell[1],
      size
    );

    const makeMatrix = (
      originX: number,
      originY: number,
      kind: 'A' | 'B' | 'C',
      outArr: THREE.Mesh[]
    ) => {
      for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
          const idx = r * size + c;
          let isHi = false;
          if (kind === 'A') isHi = rowA.includes(idx);
          if (kind === 'B') isHi = colB.includes(idx);
          if (kind === 'C') isHi = r === cellC[0] && c === cellC[1];

          const mat = new THREE.MeshStandardMaterial({
            color: isHi ? PALETTE.coral : PALETTE.navy,
            roughness: isHi ? 0.25 : 0.5,
            metalness: 0.2,
            transparent: true,
            opacity: isHi ? 1 : 0.45,
          });
          const mesh = new THREE.Mesh(cellGeo.clone(), mat);
          mesh.position.set(
            originX + c * spacing - offset,
            originY + (size - 1 - r) * spacing - offset,
            0
          );
          mesh.userData = { matrixKind: kind, row: r, col: c };
          addEdges(mesh, isHi ? PALETTE.navy : PALETTE.navySoft, 0.7);
          gridContainer.add(mesh);
          outArr.push(mesh);
          interactables.push(mesh);
        }
      }
    };

    makeMatrix(-2.6, 0, 'A', cellMeshesA);
    makeMatrix(0, 0, 'B', cellMeshesB);
    makeMatrix(2.6, 0, 'C', cellMeshesC);

    const headerY = (size * spacing) / 2 + 0.45;
    [
      { x: -2.6, title: 'Matrix A' },
      { x: 0, title: 'Matrix B' },
      { x: 2.6, title: 'C = A × B' },
    ].forEach((hdr) => {
      const { obj } = create3DLabel({ title: hdr.title, compact: true });
      obj.position.set(hdr.x, headerY, 0);
      gridContainer.add(obj);
    });

    // Connecting dot-product lines from Row A & Col B into Cell C
    const targetC = cellMeshesC[cellC[0] * size + cellC[1]];
    const linePts: number[] = [];
    if (targetC) {
      rowA.forEach((idx) => {
        const src = cellMeshesA[idx];
        if (src) {
          linePts.push(src.position.x, src.position.y, 0.14);
          linePts.push(targetC.position.x, targetC.position.y, 0.14);
        }
      });
      colB.forEach((idx) => {
        const src = cellMeshesB[idx];
        if (src) {
          linePts.push(src.position.x, src.position.y, 0.14);
          linePts.push(targetC.position.x, targetC.position.y, 0.14);
        }
      });
    }
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePts, 3));
    beamLines = new THREE.LineSegments(
      lineGeo,
      new THREE.LineBasicMaterial({ color: PALETTE.coral, transparent: true, opacity: 0.72 })
    );
    gridContainer.add(beamLines);
  };

  rebuildGrid(3, [0, 0]);
  currentSize = 3;
  currentSelected = [0, 0];

  const sceneObj: BuiltScene = {
    group,
    cameraPos: new THREE.Vector3(0, 1.4, 8.6),
    cameraTarget: new THREE.Vector3(0, 0, 0),
    enableOrbit: true,
    interactables,
    update: (elapsed) => {
      cellMeshesA.forEach((m, idx) => {
        const wave = Math.sin(elapsed * 3.2 - idx * 0.45) * 0.08;
        m.scale.setScalar(1 + wave);
      });
      cellMeshesB.forEach((m, idx) => {
        const wave = Math.cos(elapsed * 3.2 - idx * 0.45) * 0.08;
        m.scale.setScalar(1 + wave);
      });
    },
    onStateChange: (state) => {
      if (
        state.matrixSize !== currentSize ||
        state.selectedCell[0] !== currentSelected[0] ||
        state.selectedCell[1] !== currentSelected[1]
      ) {
        currentSize = state.matrixSize;
        currentSelected = [...state.selectedCell] as [number, number];
        rebuildGrid(currentSize, currentSelected);
        sceneObj.interactables = interactables;
      }
    },
    onSelect: (hit) => {
      const { row, col } = hit.object.userData as { row?: number; col?: number };
      if (typeof row === 'number' && typeof col === 'number') {
        callbacks.onSelectMatrixCell?.(row, col);
      }
    },
    dispose: () => {
      cellGeo.dispose();
      disposeObjectTree(group);
    },
  };

  return sceneObj;
}

// ============================================================================
// SLIDE 5: Performance Engine — 3D Racing Bar Chart (15x Speedup)
// ============================================================================
function buildSlide5(): BuiltScene {
  const group = new THREE.Group();
  group.add(createContactShadow(4.5));

  const floor = new THREE.Mesh(
    new THREE.BoxGeometry(7.2, 0.12, 2.8),
    new THREE.MeshStandardMaterial({ color: PALETTE.navy, roughness: 0.7 })
  );
  floor.position.set(0, -1.65, 0);
  addEdges(floor, PALETTE.coral, 0.7);
  group.add(floor);

  const singleBars: THREE.Mesh[] = [];
  const parallelBars: THREE.Mesh[] = [];
  const barGeo = new THREE.BoxGeometry(0.65, 1, 0.65);

  const plaqueSpecs = [
    { title: 'Matrix Dot', sub: '420 ms vs 28 ms' },
    { title: 'ReLU 1M', sub: '95 ms vs 7 ms' },
    { title: 'Full Epoch', sub: '12.4 s vs 0.83 s' },
  ];

  BENCHMARK_ITEMS.forEach((item, idx) => {
    const x = (idx - 1) * 2.15;

    const singleMat = new THREE.MeshStandardMaterial({
      color: PALETTE.navy,
      roughness: 0.4,
      metalness: 0.25,
    });
    const sBar = new THREE.Mesh(barGeo.clone(), singleMat);
    sBar.position.set(x - 0.44, -1.58, 0);
    sBar.userData.item = item;
    addEdges(sBar, PALETTE.navySoft, 0.85);

    const parallelMat = new THREE.MeshStandardMaterial({
      color: PALETTE.coral,
      roughness: 0.28,
      metalness: 0.2,
    });
    const pBar = new THREE.Mesh(barGeo.clone(), parallelMat);
    pBar.position.set(x + 0.44, -1.58, 0);
    pBar.userData.item = item;
    addEdges(pBar, PALETTE.navy, 0.85);

    const { obj: plaqueLabel } = create3DLabel({
      title: plaqueSpecs[idx].title,
      subtitle: plaqueSpecs[idx].sub,
      compact: true,
    });
    plaqueLabel.position.set(x, -1.95, 0.95);

    group.add(sBar, pBar, plaqueLabel);
    singleBars.push(sBar);
    parallelBars.push(pBar);
  });
  barGeo.dispose();

  const legendLabel = create3DLabel({
    badge: 'BENCHMARK LEGEND',
    title: 'Navy: Single-Thread · Coral: OpenMP + BLAS',
    compact: true,
  });
  legendLabel.obj.position.set(0, 2.75, 0);
  group.add(legendLabel.obj);

  const stageLabel = create3DLabel({
    badge: 'ENGINE EXECUTION · ~5.0s RACE',
    title: 'Stage 1/4 · Dispatching OpenMP + BLAS Kernels',
    subtitle: 'Bars climb progressively to show sequential vs parallel throughput',
    compact: true,
  });
  stageLabel.obj.position.set(0, 2.12, 0);
  group.add(stageLabel.obj);

  const setStageText = (badge: string, title: string, subtitle: string) => {
    if (stageLabel.badgeEl) stageLabel.badgeEl.textContent = badge;
    stageLabel.titleEl.textContent = title;
    if (stageLabel.subEl) stageLabel.subEl.textContent = subtitle;
  };

  // Coral finish-line beam
  const finishBeam = new THREE.Mesh(
    new THREE.BoxGeometry(6.8, 0.08, 0.14),
    new THREE.MeshStandardMaterial({
      color: PALETTE.coral,
      emissive: PALETTE.coral,
      emissiveIntensity: 0.4,
      roughness: 0.2,
    })
  );
  finishBeam.position.set(0, 1.3, 0.8);
  group.add(finishBeam);

  let raceTl: gsap.core.Timeline | null = null;
  let lastMode = '';
  let lastReplay = -1;

  const runRace = (mode: 'epoch' | 'operation') => {
    raceTl?.kill();
    const targetSingleHeights = mode === 'epoch' ? [2.1, 1.35, 3.35] : [3.2, 2.2, 2.8];
    const targetParallelHeights = targetSingleHeights.map((h) => Math.max(0.24, h / 10));

    finishBeam.position.y = 1.25;
    singleBars.forEach((b) => {
      b.scale.y = 0.05;
      b.position.y = -1.58;
    });
    parallelBars.forEach((b) => {
      b.scale.y = 0.05;
      b.position.y = -1.58;
    });

    setStageText(
      'STAGE 1 / 4 · 0.0s',
      'Dispatching Matrix Dot, ReLU 1M & Epoch Kernels',
      'OpenMP threads + CBLAS dgemm launch in parallel'
    );

    raceTl = gsap.timeline();

    // Parallel bars finish across stepped surges from 0.15s to 1.25s
    parallelBars.forEach((b, i) => {
      const h = targetParallelHeights[i];
      const startT = 0.15 + i * 0.28;
      raceTl?.to(
        b.scale,
        { y: h, duration: 0.45, ease: 'power2.out' },
        startT
      );
      raceTl?.to(
        b.position,
        { y: -1.58 + h / 2, duration: 0.45, ease: 'power2.out' },
        startT
      );
    });

    raceTl.call(
      () =>
        setStageText(
          'STAGE 2 / 4 · 1.3s',
          'OpenMP + BLAS Complete (~15× Faster)',
          'All parallel kernels finished — Single-Thread still computing'
        ),
      undefined,
      1.3
    );

    raceTl.to(finishBeam.position, { y: -1.1, duration: 0.35, ease: 'bounce.out' }, 1.35);

    // Single-thread bars rise steadily across multi-step durations up to ~5.0s
    const singleDurations = [3.6, 2.8, 4.75];
    const singleStarts = [0.25, 0.2, 0.25];
    singleBars.forEach((b, i) => {
      const h = targetSingleHeights[i];
      const dur = singleDurations[i];
      const st = singleStarts[i];
      raceTl?.to(
        b.scale,
        { y: h, duration: dur, ease: 'none' },
        st
      );
      raceTl?.to(
        b.position,
        { y: -1.58 + h / 2, duration: dur, ease: 'none' },
        st
      );
    });

    raceTl.call(
      () =>
        setStageText(
          'STAGE 3 / 4 · 3.1s',
          'Single-Thread Sequential Loops Climbing',
          'Scalar loop bottleneck remains active on large Matrix & Epoch ops'
        ),
      undefined,
      3.1
    );

    raceTl.call(
      () =>
        setStageText(
          'STAGE 4 / 4 · 5.0s COMPLETE',
          '15× Speedup Confirmed (OpenMP + OpenBLAS)',
          'Click Replay 3D Race to run the 5-second step comparison again'
        ),
      undefined,
      5.0
    );
  };

  runRace('epoch');
  lastMode = 'epoch';

  return {
    group,
    cameraPos: new THREE.Vector3(0, 2.0, 9.0),
    cameraTarget: new THREE.Vector3(0, 0.05, 0),
    enableOrbit: true,
    interactables: [...singleBars, ...parallelBars],
    update: (elapsed) => {
      group.rotation.y = Math.sin(elapsed * 0.3) * 0.1;
    },
    onStateChange: (state) => {
      if (state.benchmarkMode !== lastMode || state.replayTrigger !== lastReplay) {
        lastMode = state.benchmarkMode;
        lastReplay = state.replayTrigger;
        runRace(state.benchmarkMode);
      }
    },
    onSelect: () => {
      runRace(lastMode as 'epoch' | 'operation');
    },
    dispose: () => {
      raceTl?.kill();
      disposeObjectTree(group);
    },
  };
}

// ============================================================================
// SLIDE 6: Layers — 3D Neural Network with Signal Pulses
// ============================================================================
function buildSlide6(callbacks: SceneCallbacks): BuiltScene {
  const group = new THREE.Group();
  group.add(createContactShadow(4.4));

  const netContainer = new THREE.Group();
  group.add(netContainer);

  let currentDepth = '';
  let nodeMeshes: THREE.Mesh[] = [];
  let edgeRecords: Array<{ line: THREE.Line; a: number; b: number }> = [];
  let pulseSpheres: Array<{ mesh: THREE.Mesh; path: THREE.CatmullRomCurve3; offset: number }> = [];
  let ctx: gsap.Context | null = null;
  const pulseProgress = { t: 0 };

  const rebuildNetwork = (depth: 'shallow' | 'medium' | 'deep') => {
    ctx?.revert();
    while (netContainer.children.length > 0) {
      const c = netContainer.children[0];
      netContainer.remove(c);
      disposeObjectTree(c);
    }
    nodeMeshes = [];
    edgeRecords = [];
    pulseSpheres = [];

    const counts =
      depth === 'shallow' ? [2, 6, 1] : depth === 'medium' ? [2, 8, 5, 1] : [2, 10, 7, 4, 1];
    const xSpan = 5.4;
    const layerNodes: THREE.Mesh[][] = [];
    const nodeGeo = new THREE.IcosahedronGeometry(0.16, 1);

    counts.forEach((nCount, layerIdx) => {
      const x = -xSpan / 2 + (layerIdx / (counts.length - 1)) * xSpan;
      const colNodes: THREE.Mesh[] = [];
      const isOut = layerIdx === counts.length - 1;
      for (let i = 0; i < nCount; i++) {
        const y = ((i - (nCount - 1) / 2) * 3.1) / Math.max(4, nCount);
        const z = ((i % 2) - 0.5) * 0.45;
        const mat = new THREE.MeshStandardMaterial({
          color: isOut ? PALETTE.coral : PALETTE.navy,
          roughness: 0.3,
          metalness: 0.25,
          transparent: true,
          opacity: 1,
        });
        const m = new THREE.Mesh(nodeGeo.clone(), mat);
        m.position.set(x, y, z);
        m.userData.neuronId = nodeMeshes.length;
        netContainer.add(m);
        nodeMeshes.push(m);
        colNodes.push(m);
      }
      layerNodes.push(colNodes);
    });
    nodeGeo.dispose();

    // Connect adjacent layers
    for (let l = 0; l < layerNodes.length - 1; l++) {
      for (const src of layerNodes[l]) {
        for (const dst of layerNodes[l + 1]) {
          const lineGeo = new THREE.BufferGeometry().setFromPoints([src.position, dst.position]);
          const lineMat = new THREE.LineBasicMaterial({
            color: PALETTE.navySoft,
            transparent: true,
            opacity: 0.35,
          });
          const line = new THREE.Line(lineGeo, lineMat);
          netContainer.add(line);
          edgeRecords.push({
            line,
            a: src.userData.neuronId as number,
            b: dst.userData.neuronId as number,
          });
        }
      }
    }

    // 3 Coral signal pulses along paths through layers
    const pGeo = new THREE.SphereGeometry(0.11, 12, 12);
    const pMat = new THREE.MeshBasicMaterial({ color: PALETTE.coral });
    for (let p = 0; p < 3; p++) {
      const pts = layerNodes.map((col) => col[p % col.length].position.clone());
      const curve = new THREE.CatmullRomCurve3(pts);
      const pm = new THREE.Mesh(pGeo.clone(), pMat.clone());
      netContainer.add(pm);
      pulseSpheres.push({ mesh: pm, path: curve, offset: p * 0.33 });
    }
    pGeo.dispose();
    pMat.dispose();

    ctx = gsap.context(() => {
      gsap.fromTo(
        pulseProgress,
        { t: 0 },
        { t: 1, duration: 3, repeat: -1, ease: 'none' }
      );
    });
  };

  rebuildNetwork('medium');
  currentDepth = 'medium';

  const sceneObj: BuiltScene = {
    group,
    cameraPos: new THREE.Vector3(0, 0.8, 7.6),
    cameraTarget: new THREE.Vector3(0, 0, 0),
    enableOrbit: true,
    interactables: nodeMeshes,
    update: () => {
      pulseSpheres.forEach((ps) => {
        const u = (pulseProgress.t + ps.offset) % 1;
        ps.path.getPointAt(u, _vecA);
        ps.mesh.position.copy(_vecA);
      });
    },
    onStateChange: (state) => {
      if (state.networkDepth !== currentDepth) {
        currentDepth = state.networkDepth;
        rebuildNetwork(state.networkDepth);
        sceneObj.interactables = nodeMeshes;
      }
      const sel = state.selectedNeuronId;
      nodeMeshes.forEach((m) => {
        const id = m.userData.neuronId as number;
        const mat = m.material as THREE.MeshStandardMaterial;
        if (sel === null) {
          mat.opacity = 1;
          m.scale.setScalar(1);
        } else {
          const connected =
            id === sel || edgeRecords.some((e) => (e.a === sel && e.b === id) || (e.b === sel && e.a === id));
          mat.opacity = connected ? 1 : 0.25;
          m.scale.setScalar(id === sel ? 1.45 : connected ? 1.1 : 0.85);
        }
      });
      edgeRecords.forEach((e) => {
        const mat = e.line.material as THREE.LineBasicMaterial;
        if (sel === null) {
          mat.color.setHex(PALETTE.navySoft);
          mat.opacity = 0.35;
        } else if (e.a === sel || e.b === sel) {
          mat.color.setHex(PALETTE.coral);
          mat.opacity = 0.95;
        } else {
          mat.color.setHex(PALETTE.navySoft);
          mat.opacity = 0.08;
        }
      });
    },
    onSelect: (hit) => {
      const id = hit.object.userData.neuronId as number;
      if (typeof id === 'number') {
        callbacks.onSelectNeuron?.(id);
      }
    },
    dispose: () => {
      ctx?.revert();
      disposeObjectTree(group);
    },
  };

  return sceneObj;
}

// ============================================================================
// SLIDE 7: Activations & Losses (ReLU surface + MSE parabolic bowl)
// ============================================================================
function buildSlide7(callbacks: SceneCallbacks): BuiltScene {
  const group = new THREE.Group();
  group.add(createContactShadow(4.5));

  // Left: ReLU / LeakyReLU Surface
  const reluGeo = new THREE.PlaneGeometry(3.1, 3.1, 20, 20);
  reluGeo.rotateX(-Math.PI / 2);
  const reluMat = new THREE.MeshStandardMaterial({
    color: PALETTE.navy,
    roughness: 0.42,
    metalness: 0.2,
    side: THREE.DoubleSide,
  });
  const reluMesh = new THREE.Mesh(reluGeo, reluMat);
  reluMesh.position.set(-2.15, -0.35, 0);
  const reluWire = new THREE.LineSegments(
    new THREE.WireframeGeometry(reluGeo),
    new THREE.LineBasicMaterial({ color: PALETTE.coral, transparent: true, opacity: 0.55 })
  );
  reluMesh.add(reluWire);
  group.add(reluMesh);

  // Right: MSE Parabolic Bowl
  const mseGeo = new THREE.PlaneGeometry(3.1, 3.1, 20, 20);
  mseGeo.rotateX(-Math.PI / 2);
  const msePos = mseGeo.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < msePos.count; i++) {
    const x = msePos.getX(i);
    const z = msePos.getZ(i);
    const y = (x * x + z * z) * 0.45;
    msePos.setY(i, y);
  }
  mseGeo.computeVertexNormals();
  const mseMat = new THREE.MeshStandardMaterial({
    color: PALETTE.coral,
    roughness: 0.35,
    metalness: 0.18,
    side: THREE.DoubleSide,
  });
  const mseMesh = new THREE.Mesh(mseGeo, mseMat);
  mseMesh.position.set(2.15, -0.55, 0);
  mseMesh.userData.kind = 'mse';
  const mseWire = new THREE.LineSegments(
    new THREE.WireframeGeometry(mseGeo),
    new THREE.LineBasicMaterial({ color: PALETTE.navy, transparent: true, opacity: 0.5 })
  );
  mseMesh.add(mseWire);
  group.add(mseMesh);

  // Coral loss sample marker inside MSE bowl
  const sampleMarker = new THREE.Mesh(
    new THREE.SphereGeometry(0.16, 16, 16),
    new THREE.MeshStandardMaterial({
      color: PALETTE.navy,
      emissive: PALETTE.coral,
      emissiveIntensity: 0.5,
    })
  );
  mseMesh.add(sampleMarker);

  const sampleCoords = [
    [0.0, 0.0],
    [0.8, -0.5],
    [-0.9, 0.7],
    [1.1, 0.8],
  ];

  const updateReluSurface = (alpha: number, morphFactor: number) => {
    const pos = reluGeo.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = evaluateLeakyRelu(x, alpha) * morphFactor * 0.9;
      pos.setY(i, y);
    }
    pos.needsUpdate = true;
    reluGeo.computeVertexNormals();
    reluWire.geometry.dispose();
    reluWire.geometry = new THREE.WireframeGeometry(reluGeo);
  };

  const animState = { morph: 0 };
  const ctx = gsap.context(() => {
    gsap.to(animState, {
      morph: 1,
      duration: 1.0,
      ease: 'power2.out',
      onUpdate: () => updateReluSurface(0, animState.morph),
    });
  });

  let currentAlpha = 0;
  let currentSample = 0;
  const setMarker = (idx: number) => {
    const [sx, sz] = sampleCoords[idx % sampleCoords.length];
    const sy = (sx * sx + sz * sz) * 0.45 + 0.16;
    gsap.to(sampleMarker.position, { x: sx, y: sy, z: sz, duration: 0.45, ease: 'back.out(1.7)' });
  };
  setMarker(0);

  return {
    group,
    cameraPos: new THREE.Vector3(0, 2.7, 6.8),
    cameraTarget: new THREE.Vector3(0, 0, 0),
    enableOrbit: true,
    interactables: [mseMesh],
    update: (_elapsed, delta) => {
      mseMesh.rotation.y += delta * 0.3;
    },
    onStateChange: (state) => {
      if (state.reluAlpha !== currentAlpha) {
        currentAlpha = state.reluAlpha;
        updateReluSurface(currentAlpha, 1);
      }
      if (state.mseSampleIndex !== currentSample) {
        currentSample = state.mseSampleIndex;
        setMarker(currentSample);
      }
    },
    onSelect: () => {
      callbacks.onSelectMseSample?.((currentSample + 1) % sampleCoords.length);
    },
    dispose: () => {
      ctx.revert();
      disposeObjectTree(group);
    },
  };
}

// ============================================================================
// SLIDE 8: Optimizers (SGD, Adagrad, RMSprop, Adam)
// ============================================================================
function buildSlide8(callbacks: SceneCallbacks): BuiltScene {
  const group = new THREE.Group();
  group.add(createContactShadow(4.5));

  const names: OptimizerName[] = ['SGD', 'Adagrad', 'RMSprop', 'Adam'];
  const chipMeshes: THREE.Mesh[] = [];
  const trails: THREE.Mesh[] = [];
  const particles: THREE.Mesh[] = [];
  const labelCards: LabelCardRefs[] = [];
  const trajectories = names.map((n) =>
    generateOptimizerTrajectory(n, 32).map((p) => new THREE.Vector3(p.x, p.y + 1.1, p.z))
  );

  const chipGeo = new THREE.BoxGeometry(1.45, 0.56, 0.55);
  const pGeo = new THREE.SphereGeometry(0.11, 14, 14);

  names.forEach((name, idx) => {
    const x = (idx - 1.5) * 1.85;
    const holder = new THREE.Group();
    holder.position.set(x, -0.65, 0);

    const chip = new THREE.Mesh(
      chipGeo.clone(),
      new THREE.MeshStandardMaterial({ color: PALETTE.navy, roughness: 0.4, metalness: 0.22 })
    );
    chip.userData.optimizer = name;
    addEdges(chip, PALETTE.coral, 0.85);

    const optLabel = create3DLabel({
      title: name,
      subtitle: 'Optimizer',
      compact: true,
      onClick: () => callbacks.onSelectOptimizer?.(name),
    });
    optLabel.obj.position.set(0, -0.68, 0.28);

    holder.add(chip, optLabel.obj);
    chipMeshes.push(chip);
    labelCards.push(optLabel);

    // Trajectory TubeGeometry
    const curve = new THREE.CatmullRomCurve3(trajectories[idx]);
    const tubeGeo = new THREE.TubeGeometry(curve, 36, 0.032, 8, false);
    const tubeMat = new THREE.MeshBasicMaterial({
      color: PALETTE.coral,
      transparent: true,
      opacity: name === 'Adam' ? 0.95 : 0.28,
    });
    const tube = new THREE.Mesh(tubeGeo, tubeMat);
    holder.add(tube);
    trails.push(tube);

    // Momentum particle
    const particle = new THREE.Mesh(
      pGeo.clone(),
      new THREE.MeshStandardMaterial({
        color: PALETTE.coral,
        emissive: PALETTE.coral,
        emissiveIntensity: 0.45,
      })
    );
    holder.add(particle);
    particles.push(particle);

    group.add(holder);
  });

  chipGeo.dispose();
  pGeo.dispose();

  const phase = { t: 0 };
  const ctx = gsap.context(() => {
    gsap.to(phase, { t: 1, duration: 3.6, repeat: -1, ease: 'none' });
  });

  const curves = trajectories.map((pts) => new THREE.CatmullRomCurve3(pts));
  let selected: OptimizerName = 'Adam';

  return {
    group,
    cameraPos: new THREE.Vector3(0, 1.6, 8.8),
    cameraTarget: new THREE.Vector3(0, 0.1, 0),
    enableOrbit: false,
    interactables: chipMeshes,
    update: () => {
      names.forEach((name, idx) => {
        if (name === selected || selected === undefined) {
          const u = (phase.t + idx * 0.2) % 1;
          curves[idx].getPointAt(u, _vecA);
          particles[idx].position.copy(_vecA);
        }
      });
    },
    onStateChange: (state) => {
      selected = state.selectedOptimizer;
      names.forEach((name, idx) => {
        const isSel = name === selected;
        const tubeMat = trails[idx].material as THREE.MeshBasicMaterial;
        tubeMat.opacity = isSel ? 0.95 : 0.18;
        gsap.to(chipMeshes[idx].scale, {
          x: isSel ? 1.15 : 0.95,
          y: isSel ? 1.15 : 0.95,
          z: isSel ? 1.15 : 0.95,
          duration: 0.3,
        });
        set3DLabelActive(labelCards[idx], isSel);
        labelCards[idx].el.style.opacity = isSel ? '1' : '0.72';
      });
    },
    onSelect: (hit) => {
      const opt = hit.object.userData.optimizer as OptimizerName;
      if (opt) callbacks.onSelectOptimizer?.(opt);
    },
    dispose: () => {
      ctx.revert();
      disposeObjectTree(group);
    },
  };
}

// ============================================================================
// SLIDE 9: Model & Training Loop (Forward -> Backward -> Update)
// ============================================================================
function buildSlide9(callbacks: SceneCallbacks): BuiltScene {
  const group = new THREE.Group();
  group.add(createContactShadow(4.0));

  const stages: Array<{ label: 'Forward' | 'Backward' | 'Update'; pos: [number, number, number] }> = [
    { label: 'Forward', pos: [-2.1, -0.4, 1.1] },
    { label: 'Backward', pos: [2.1, -0.4, 1.1] },
    { label: 'Update', pos: [0, -0.4, -2.0] },
  ];

  const pillars: THREE.Mesh[] = [];
  const labelCards: LabelCardRefs[] = [];
  stages.forEach((st, idx) => {
    const pillar = new THREE.Mesh(
      new THREE.BoxGeometry(1.35, 1.75, 1.35),
      new THREE.MeshStandardMaterial({ color: PALETTE.navy, roughness: 0.42, metalness: 0.22 })
    );
    pillar.position.set(...st.pos);
    pillar.userData.stage = st.label;
    addEdges(pillar, PALETTE.coral, 0.85);

    const stageLabel = create3DLabel({
      badge: `0${idx + 1}`,
      title: st.label,
      compact: true,
      onClick: () => callbacks.onSelectLoopStage?.(activeFilter === st.label ? 'all' : st.label),
    });
    stageLabel.obj.position.set(0, 1.45, 0);
    pillar.add(stageLabel.obj);

    group.add(pillar);
    pillars.push(pillar);
    labelCards.push(stageLabel);
  });

  const loopCurve = new THREE.CatmullRomCurve3(
    [
      new THREE.Vector3(-2.1, 0.85, 1.1),
      new THREE.Vector3(2.1, 0.85, 1.1),
      new THREE.Vector3(0, 0.85, -2.0),
    ],
    true
  );
  const rail = new THREE.Mesh(
    new THREE.TubeGeometry(loopCurve, 48, 0.045, 8, true),
    new THREE.MeshBasicMaterial({ color: PALETTE.coral })
  );
  group.add(rail);

  const batchSphere = new THREE.Mesh(
    new THREE.SphereGeometry(0.24, 18, 18),
    new THREE.MeshStandardMaterial({
      color: PALETTE.coral,
      emissive: PALETTE.coral,
      emissiveIntensity: 0.45,
      roughness: 0.2,
    })
  );
  group.add(batchSphere);

  const loopProgress = { t: 0 };
  const tween = gsap.to(loopProgress, {
    t: 1,
    duration: 4,
    repeat: -1,
    ease: 'none',
  });

  let activeFilter: 'all' | 'Forward' | 'Backward' | 'Update' = 'all';

  return {
    group,
    cameraPos: new THREE.Vector3(0, 3.8, 7.6),
    cameraTarget: new THREE.Vector3(0, -0.1, -0.2),
    enableOrbit: true,
    interactables: pillars,
    update: () => {
      loopCurve.getPointAt(loopProgress.t % 1, _vecA);
      batchSphere.position.copy(_vecA);
      pillars.forEach((p) => {
        const dist = p.position.distanceTo(_vecA);
        const pulse = dist < 1.55 ? 1.16 : 1.0;
        const isStageSel = activeFilter === p.userData.stage;
        p.scale.setScalar(isStageSel ? 1.24 : pulse);
      });
    },
    onStateChange: (state) => {
      activeFilter = state.loopStage;
      pillars.forEach((p, idx) => {
        const isStageSel = activeFilter === p.userData.stage;
        set3DLabelActive(labelCards[idx], isStageSel);
      });
      if (state.loopStage === 'all') {
        tween.resume();
        tween.timeScale(state.loopSpeed);
      } else {
        tween.pause();
        const stageIdx = stages.findIndex((s) => s.label === state.loopStage);
        if (stageIdx >= 0) {
          loopProgress.t = stageIdx / 3;
        }
      }
    },
    onSelect: (hit) => {
      const stage = hit.object.userData.stage as 'Forward' | 'Backward' | 'Update';
      if (stage) {
        callbacks.onSelectLoopStage?.(activeFilter === stage ? 'all' : stage);
      }
    },
    dispose: () => {
      tween.kill();
      disposeObjectTree(group);
    },
  };
}

// ============================================================================
// SLIDE 10: Datasets & StandardScaler Pipeline
// ============================================================================
function buildSlide10(callbacks: SceneCallbacks): BuiltScene {
  const group = new THREE.Group();
  group.add(createContactShadow(4.6));

  // Raw Dataset Cylinder (left)
  const rawCyl = new THREE.Mesh(
    new THREE.CylinderGeometry(0.85, 0.85, 1.9, 18),
    new THREE.MeshStandardMaterial({ color: PALETTE.navy, roughness: 0.4, metalness: 0.25 })
  );
  rawCyl.position.set(-2.7, 0, 0);
  rawCyl.userData.role = 'raw';
  addEdges(rawCyl, PALETTE.coral, 0.8);

  // StandardScaler Cube (center)
  const scalerCube = new THREE.Mesh(
    new THREE.BoxGeometry(1.5, 1.5, 1.5),
    new THREE.MeshStandardMaterial({
      color: PALETTE.coral,
      roughness: 0.25,
      metalness: 0.15,
      transparent: true,
      opacity: 0.88,
    })
  );
  scalerCube.position.set(0, 0, 0);
  scalerCube.userData.role = 'scaler';
  addEdges(scalerCube, PALETTE.navy, 0.9);

  // Scaled Train/Test Plane (right)
  const outPlane = new THREE.Mesh(
    new THREE.BoxGeometry(1.8, 0.18, 1.8),
    new THREE.MeshStandardMaterial({ color: PALETTE.navy, roughness: 0.45 })
  );
  outPlane.position.set(2.7, -0.55, 0);
  addEdges(outPlane, PALETTE.coral, 0.85);

  group.add(rawCyl, scalerCube, outPlane);

  // InstancedMesh sample spheres streaming through the pipeline
  const sampleCount = 36;
  const sGeo = new THREE.SphereGeometry(0.09, 10, 10);
  const sMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3 });
  const samples = new THREE.InstancedMesh(sGeo, sMat, sampleCount);
  samples.frustumCulled = false;
  group.add(samples);

  const offsets = Array.from({ length: sampleCount }, (_, i) => ({
    phase: i / sampleCount,
    yJitter: ((i % 7) - 3) * 0.14,
    zJitter: (((i * 3) % 5) - 2) * 0.16,
  }));

  const resample = () => {
    offsets.forEach((o, i) => {
      o.yJitter = Math.sin(i * 1.9 + performance.now() * 0.002) * 0.55;
      o.zJitter = Math.cos(i * 2.3 + performance.now() * 0.002) * 0.45;
    });
  };

  let lastResample = -1;
  let lastFit = false;

  return {
    group,
    cameraPos: new THREE.Vector3(0, 2.2, 8.6),
    cameraTarget: new THREE.Vector3(0, -0.1, 0),
    enableOrbit: false,
    interactables: [rawCyl, scalerCube],
    update: (elapsed) => {
      offsets.forEach((o, i) => {
        const u = (elapsed * 0.18 + o.phase) % 1;
        const x = THREE.MathUtils.lerp(-2.7, 2.7, u);
        // After passing scaler cube (x > 0), variance shrinks to normalized band
        const spread = x < 0 ? 1.0 : 0.35;
        const y = o.yJitter * spread + Math.sin(elapsed * 2 + i) * 0.06;
        const z = o.zJitter * spread;
        _dummy.position.set(x, y, z);
        _dummy.scale.setScalar(x > 0 ? 0.95 : 1.15);
        _dummy.updateMatrix();
        samples.setMatrixAt(i, _dummy.matrix);
        _color.setHex(x > 0 ? PALETTE.coral : PALETTE.navy);
        samples.setColorAt(i, _color);
      });
      samples.instanceMatrix.needsUpdate = true;
      if (samples.instanceColor) samples.instanceColor.needsUpdate = true;
      scalerCube.rotation.y = elapsed * 0.45;
    },
    onStateChange: (state) => {
      if (state.resampleTrigger !== lastResample) {
        lastResample = state.resampleTrigger;
        resample();
      }
      if (state.scalerFitMode !== lastFit) {
        lastFit = state.scalerFitMode;
        gsap.to(scalerCube.scale, {
          x: lastFit ? 1.38 : 1,
          y: lastFit ? 1.38 : 1,
          z: lastFit ? 1.38 : 1,
          duration: 0.45,
          ease: 'back.out(1.7)',
        });
      }
    },
    onSelect: (hit) => {
      const role = hit.object.userData.role;
      if (role === 'raw') callbacks.onResampleData?.();
      if (role === 'scaler') callbacks.onToggleFitMode?.();
    },
    dispose: () => {
      disposeObjectTree(group);
    },
  };
}

// ============================================================================
// SLIDE 11: Examples — Neural SVM & Navier-Stokes 2D
// ============================================================================
function buildSlide11(callbacks: SceneCallbacks): BuiltScene {
  const group = new THREE.Group();
  group.add(createContactShadow(4.8));

  // Left Group: Neural SVM 3D Scatter + Decision Plane
  const svmGroup = new THREE.Group();
  svmGroup.position.set(-1.95, 0, 0);
  group.add(svmGroup);

  const ptGeo = new THREE.CylinderGeometry(0.11, 0.11, 0.36, 10);
  const ptMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.35 });
  const svmPoints = new THREE.InstancedMesh(ptGeo, ptMat, 32);
  svmPoints.frustumCulled = false;

  for (let i = 0; i < 32; i++) {
    const cls = i < 16 ? 1 : -1;
    const angle = (i % 16) * 0.4;
    const r = 0.55 + ((i * 7) % 5) * 0.14;
    const px = Math.cos(angle) * r + cls * 0.45;
    const py = cls * 0.55 + Math.sin(i) * 0.25;
    const pz = Math.sin(angle) * r;
    _dummy.position.set(px, py, pz);
    _dummy.scale.setScalar(1);
    _dummy.updateMatrix();
    svmPoints.setMatrixAt(i, _dummy.matrix);
    _color.setHex(cls === 1 ? PALETTE.coral : PALETTE.navy);
    svmPoints.setColorAt(i, _color);
  }
  svmPoints.instanceMatrix.needsUpdate = true;
  if (svmPoints.instanceColor) svmPoints.instanceColor.needsUpdate = true;
  svmPoints.computeBoundingSphere();
  svmGroup.add(svmPoints);

  const planeMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(2.6, 2.6),
    new THREE.MeshStandardMaterial({
      color: PALETTE.coral,
      transparent: true,
      opacity: 0.38,
      side: THREE.DoubleSide,
      roughness: 0.2,
    })
  );
  planeMesh.rotation.x = Math.PI / 2 - 0.35;
  planeMesh.userData.kind = 'svmPlane';
  addEdges(planeMesh, PALETTE.coral, 0.95);
  svmGroup.add(planeMesh);

  // Right Group: Navier-Stokes 2D Vorticity Grid
  const nsGroup = new THREE.Group();
  nsGroup.position.set(1.95, 0, 0);
  group.add(nsGroup);

  const gridN = 8;
  const cellGeo = new THREE.BoxGeometry(0.28, 0.28, 0.28);
  const cellMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3 });
  const nsCells = new THREE.InstancedMesh(cellGeo, cellMat, gridN * gridN);
  nsCells.frustumCulled = false;
  nsGroup.add(nsCells);

  let planeSnapped = false;
  let lastSnap = -1;
  const planeSweep = gsap.to(planeMesh.rotation, {
    z: 0.5,
    duration: 4.5,
    repeat: -1,
    yoyo: true,
    ease: 'sine.inOut',
  });

  const snapPlane = () => {
    planeSnapped = true;
    planeSweep.pause();
    gsap.to(planeMesh.rotation, { x: Math.PI / 2 - 0.18, z: 0, duration: 0.55, ease: 'power2.out' });
  };

  return {
    group,
    cameraPos: new THREE.Vector3(0, 2.4, 8.8),
    cameraTarget: new THREE.Vector3(0, 0, 0),
    enableOrbit: true,
    interactables: [planeMesh],
    update: (elapsed) => {
      let idx = 0;
      for (let gx = 0; gx < gridN; gx++) {
        for (let gy = 0; gy < gridN; gy++) {
          const x = (gx - (gridN - 1) / 2) * 0.34;
          const y = (gy - (gridN - 1) / 2) * 0.34;
          const w = sampleNavierStokesVorticity(x, y, elapsed * 0.8);
          _dummy.position.set(x, y * 0.75, w * 0.35);
          _dummy.scale.set(1, 1, 0.8 + Math.abs(w) * 0.9);
          _dummy.updateMatrix();
          nsCells.setMatrixAt(idx, _dummy.matrix);
          if (w >= 0) {
            _color.setHex(PALETTE.coral).lerp(new THREE.Color(PALETTE.cream), Math.max(0, 0.45 - w * 0.35));
          } else {
            _color.setHex(PALETTE.navy);
          }
          nsCells.setColorAt(idx, _color);
          idx++;
        }
      }
      nsCells.instanceMatrix.needsUpdate = true;
      if (nsCells.instanceColor) nsCells.instanceColor.needsUpdate = true;
      if (!planeSnapped) {
        svmGroup.rotation.y = Math.sin(elapsed * 0.4) * 0.22;
      }
    },
    onStateChange: (state, camera) => {
      if (state.snapSvmPlaneTrigger !== lastSnap && lastSnap !== -1) {
        snapPlane();
      }
      lastSnap = state.snapSvmPlaneTrigger;
      const targetX = state.exampleFocus === 'svm' ? -1.8 : state.exampleFocus === 'navier' ? 1.8 : 0;
      const targetZ = state.exampleFocus === 'both' ? 8.8 : 5.4;
      gsap.to(camera.position, { x: targetX, z: targetZ, duration: 0.55, ease: 'power2.out' });
    },
    onSelect: () => {
      snapPlane();
      callbacks.onSnapSvm?.();
    },
    dispose: () => {
      planeSweep.kill();
      disposeObjectTree(group);
    },
  };
}

// ============================================================================
// SLIDE 12: Extending CppTorch (3-Step Workflow)
// ============================================================================
function buildSlide12(callbacks: SceneCallbacks): BuiltScene {
  const group = new THREE.Group();
  group.add(createContactShadow(4.2));

  const steps = ['1 · Create Header', '2 · Register Factory', '3 · Export cpptorch.h'];
  const stepBlocks: THREE.Group[] = [];
  const hitMeshes: THREE.Mesh[] = [];
  const badges: THREE.Mesh[] = [];
  const labelCards: LabelCardRefs[] = [];

  steps.forEach((label, idx) => {
    const sg = new THREE.Group();
    sg.position.set(0, (1 - idx) * 1.35, 0);

    const block = new THREE.Mesh(
      new THREE.BoxGeometry(3.2, 0.82, 0.65),
      new THREE.MeshStandardMaterial({ color: PALETTE.navy, roughness: 0.4, metalness: 0.2 })
    );
    block.userData.step = idx;
    addEdges(block, PALETTE.coral, 0.85);

    const stepLabel = create3DLabel({
      badge: `Step 0${idx + 1}`,
      title: label,
      compact: true,
      onClick: () => callbacks.onSelectStep?.(idx),
    });
    stepLabel.obj.position.set(0, 0, 0.42);

    const badge = new THREE.Mesh(
      new THREE.BoxGeometry(0.46, 0.46, 0.46),
      new THREE.MeshStandardMaterial({ color: PALETTE.coral, roughness: 0.25 })
    );
    badge.position.set(-2.05, 0, 0);
    badge.userData.step = idx;
    addEdges(badge, PALETTE.navy, 0.9);

    sg.add(block, badge, stepLabel.obj);
    group.add(sg);
    stepBlocks.push(sg);
    hitMeshes.push(block, badge);
    badges.push(badge);
    labelCards.push(stepLabel);
  });

  // Connector lines
  const connGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-2.05, 1.35, 0),
    new THREE.Vector3(-2.05, -1.35, 0),
  ]);
  const conn = new THREE.Line(connGeo, new THREE.LineBasicMaterial({ color: PALETTE.coral }));
  group.add(conn);

  const ctx = gsap.context(() => {
    gsap.from(
      stepBlocks.map((b) => b.scale),
      { x: 0.1, y: 0.1, z: 0.1, duration: 0.65, stagger: 0.2, ease: 'back.out(1.7)' }
    );
    badges.forEach((b, i) => {
      gsap.to(b.position, {
        y: 0.14,
        duration: 1.4,
        delay: i * 0.2,
        yoyo: true,
        repeat: -1,
        ease: 'sine.inOut',
      });
    });
  });

  return {
    group,
    cameraPos: new THREE.Vector3(0, 0.6, 7.2),
    cameraTarget: new THREE.Vector3(0, 0, 0),
    enableOrbit: false,
    interactables: hitMeshes,
    update: (elapsed) => {
      group.rotation.y = Math.sin(elapsed * 0.45) * 0.14;
    },
    onStateChange: (state) => {
      stepBlocks.forEach((sg, idx) => {
        const sel = state.selectedStep;
        const scale = sel === null ? 1 : sel === idx ? 1.2 : 0.86;
        gsap.to(sg.scale, { x: scale, y: scale, z: scale, duration: 0.35, ease: 'power2.out' });
        const isSel = sel === idx;
        set3DLabelActive(labelCards[idx], isSel);
        labelCards[idx].el.style.opacity = sel === null || isSel ? '1' : '0.5';
      });
    },
    onSelect: (hit) => {
      const step = hit.object.userData.step as number;
      if (typeof step === 'number') callbacks.onSelectStep?.(step);
    },
    dispose: () => {
      ctx.revert();
      disposeObjectTree(group);
    },
  };
}

// ============================================================================
// SLIDE 13: Thank You Emblem & Author Rings
// ============================================================================
function buildSlide13(callbacks: SceneCallbacks): BuiltScene {
  const group = new THREE.Group();
  group.add(createContactShadow(3.6));

  const emblem = new THREE.Group();
  group.add(emblem);

  // Central neural cluster
  const core = new THREE.Mesh(
    new THREE.OctahedronGeometry(0.75, 1),
    new THREE.MeshStandardMaterial({ color: PALETTE.navy, roughness: 0.3, metalness: 0.35 })
  );
  addEdges(core, PALETTE.coral, 0.9);
  emblem.add(core);

  const ringTilts = [
    [0.55, 0.2, 0],
    [-0.48, 0.6, 0.4],
    [0.15, -0.5, 0.85],
  ];
  const rings: THREE.Mesh[] = [];
  ringTilts.forEach((rot, idx) => {
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(1.55 + idx * 0.28, 0.045, 10, 48),
      new THREE.MeshStandardMaterial({
        color: idx === 1 ? PALETTE.coral : PALETTE.navy,
        roughness: 0.3,
        metalness: 0.3,
      })
    );
    ring.rotation.set(rot[0], rot[1], rot[2]);
    ring.userData.authorIdx = idx;
    group.add(ring);
    rings.push(ring);
  });

  // Burst particles (distinct color per selected team member)
  const MEMBER_BURST_COLORS = [0x2563eb, 0x0d9488, 0xd97706];
  const burstCount = 32;
  const bGeo = new THREE.BoxGeometry(0.16, 0.16, 0.16);
  const bMat = new THREE.MeshStandardMaterial({ color: PALETTE.coral, roughness: 0.25 });
  const burstMesh = new THREE.InstancedMesh(bGeo, bMat, burstCount);
  burstMesh.frustumCulled = false;
  group.add(burstMesh);

  const dirs = Array.from({ length: burstCount }, (_, i) => {
    const phi = Math.acos(1 - (2 * (i + 0.5)) / burstCount);
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;
    return new THREE.Vector3(
      Math.sin(phi) * Math.cos(theta),
      Math.sin(phi) * Math.sin(theta),
      Math.cos(phi)
    );
  });

  const burstState = { radius: 1.9, scale: 0.45 };
  const ctx = gsap.context(() => {
    gsap.to(emblem.rotation, { y: Math.PI * 2, duration: 10, repeat: -1, ease: 'none' });
  });

  const triggerBurst = (colorHex = PALETTE.coral) => {
    bMat.color.setHex(colorHex);
    gsap.fromTo(
      burstState,
      { radius: 0.6, scale: 1.55 },
      { radius: 2.35, scale: 0.55, duration: 1.45, ease: 'power2.out' }
    );
  };

  let lastBurst = -1;
  let initializedAuthor = false;
  let lastAuthorIdx: number | null = null;

  return {
    group,
    cameraPos: new THREE.Vector3(0, 0.4, 6.4),
    cameraTarget: new THREE.Vector3(0, 0, 0),
    enableOrbit: false,
    interactables: [core, ...rings],
    update: (elapsed, delta) => {
      rings.forEach((r, i) => {
        r.rotation.z += delta * (0.35 + i * 0.18);
      });
      dirs.forEach((d, i) => {
        _dummy.position.copy(d).multiplyScalar(burstState.radius + Math.sin(elapsed * 2 + i) * 0.08);
        _dummy.scale.setScalar(burstState.scale);
        _dummy.rotation.set(elapsed + i, elapsed * 0.7, 0);
        _dummy.updateMatrix();
        burstMesh.setMatrixAt(i, _dummy.matrix);
      });
      burstMesh.instanceMatrix.needsUpdate = true;
    },
    onStateChange: (state) => {
      const activeColor =
        state.selectedAuthorIndex !== null
          ? MEMBER_BURST_COLORS[state.selectedAuthorIndex] ?? PALETTE.coral
          : PALETTE.coral;

      if (lastBurst !== -1 && state.burstTrigger !== lastBurst) {
        triggerBurst(activeColor);
      } else if (
        initializedAuthor &&
        state.selectedAuthorIndex !== lastAuthorIdx &&
        state.selectedAuthorIndex !== null
      ) {
        triggerBurst(activeColor);
      }

      lastBurst = state.burstTrigger;
      lastAuthorIdx = state.selectedAuthorIndex;
      initializedAuthor = true;

      rings.forEach((r, idx) => {
        const sel = state.selectedAuthorIndex === idx;
        (r.material as THREE.MeshStandardMaterial).color.setHex(
          sel ? MEMBER_BURST_COLORS[idx] ?? PALETTE.coral : idx === 1 ? PALETTE.coral : PALETTE.navy
        );
        r.scale.setScalar(sel ? 1.12 : 1.0);
      });
    },
    onSelect: (hit) => {
      const aIdx = hit.object.userData.authorIdx;
      if (typeof aIdx === 'number') {
        callbacks.onSelectAuthor?.(aIdx);
      } else {
        const activeColor =
          lastAuthorIdx !== null
            ? MEMBER_BURST_COLORS[lastAuthorIdx] ?? PALETTE.coral
            : PALETTE.coral;
        triggerBurst(activeColor);
      }
    },
    dispose: () => {
      ctx.revert();
      disposeObjectTree(group);
    },
  };
}

export function buildSlideScene(slideId: number, callbacks: SceneCallbacks): BuiltScene {
  switch (slideId) {
    case 1:
      return buildSlide1(callbacks);
    case 2:
      return buildSlide2(callbacks);
    case 3:
      return buildSlide3(callbacks);
    case 4:
      return buildSlide4(callbacks);
    case 5:
      return buildSlide5();
    case 6:
      return buildSlide6(callbacks);
    case 7:
      return buildSlide7(callbacks);
    case 8:
      return buildSlide8(callbacks);
    case 9:
      return buildSlide9(callbacks);
    case 10:
      return buildSlide10(callbacks);
    case 11:
      return buildSlide11(callbacks);
    case 12:
      return buildSlide12(callbacks);
    case 13:
    default:
      return buildSlide13(callbacks);
  }
}
