import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import {
  buildSlideScene,
  PALETTE,
  type BuiltScene,
  type SceneCallbacks,
  type SlideSceneState,
} from '../three/sceneBuilders';

interface ThreeViewportProps {
  slideId: number;
  sceneState: SlideSceneState;
  callbacks: SceneCallbacks;
  interactionHint: string;
}

function detectSoftwareGl(): boolean {
  try {
    const probeCanvas = document.createElement('canvas');
    const gl = probeCanvas.getContext('webgl2');
    if (!gl) return true;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
    const gpu = String(gl.getParameter(param) || '');
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
  } catch {
    return true;
  }
}

export function ThreeViewport({
  slideId,
  sceneState,
  callbacks,
  interactionHint,
}: ThreeViewportProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const labelMountRef = useRef<HTMLDivElement | null>(null);
  const [contextLost, setContextLost] = useState(false);

  const stateRef = useRef(sceneState);
  stateRef.current = sceneState;

  const callbacksRef = useRef(callbacks);
  callbacksRef.current = callbacks;

  const activeSceneRef = useRef<BuiltScene | null>(null);
  const sceneRootRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const labelRendererRef = useRef<CSS2DRenderer | null>(null);

  // Initialize WebGLRenderer & CSS2DRenderer once
  useEffect(() => {
    const container = mountRef.current;
    const labelContainer = labelMountRef.current;
    if (!container || !labelContainer) return;

    const isSoftwareGl = detectSoftwareGl();
    const width = Math.max(240, container.clientWidth);
    const height = Math.max(240, container.clientHeight);

    const renderer = new THREE.WebGLRenderer({
      antialias: !isSoftwareGl,
      powerPreference: 'high-performance',
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height, false);
    renderer.shadowMap.enabled = !isSoftwareGl;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';

    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(width, height);
    labelRenderer.domElement.style.position = 'absolute';
    labelRenderer.domElement.style.top = '0';
    labelRenderer.domElement.style.left = '0';
    labelRenderer.domElement.style.pointerEvents = 'none';
    labelContainer.innerHTML = '';
    labelContainer.appendChild(labelRenderer.domElement);
    labelRendererRef.current = labelRenderer;

    const handleContextLost = (e: Event) => {
      e.preventDefault();
      setContextLost(true);
    };
    const handleContextRestored = () => {
      setContextLost(false);
    };
    renderer.domElement.addEventListener('webglcontextlost', handleContextLost);
    renderer.domElement.addEventListener('webglcontextrestored', handleContextRestored);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(PALETTE.cream);
    scene.fog = new THREE.FogExp2(PALETTE.cream, 0.022);
    sceneRootRef.current = scene;

    const pmrem = new THREE.PMREMGenerator(renderer);
    const envRT = pmrem.fromScene(new RoomEnvironment(), 0.04);
    scene.environment = envRT.texture;
    pmrem.dispose();

    const hemiLight = new THREE.HemisphereLight(0xf6f8ff, 0xd9ceba, 1.15);
    scene.add(hemiLight);

    const keyLight = new THREE.DirectionalLight(0xfff5e8, 1.65);
    keyLight.position.set(5, 8, 6);
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0xffb4a8, 0.7);
    rimLight.position.set(-6, 4, -5);
    scene.add(rimLight);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 1.2, 6.8);
    cameraRef.current = camera;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = !isSoftwareGl;
    controls.dampingFactor = 0.07;
    controls.enablePan = false;
    controls.minDistance = 3.5;
    controls.maxDistance = 12;
    controlsRef.current = controls;

    // Raycasting on click + throttled hover cursor
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let pointerDownPos = { x: 0, y: 0 };

    const updatePointer = (ev: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    };

    const onPointerDown = (ev: PointerEvent) => {
      pointerDownPos = { x: ev.clientX, y: ev.clientY };
    };

    const onPointerUp = (ev: PointerEvent) => {
      const dist = Math.hypot(ev.clientX - pointerDownPos.x, ev.clientY - pointerDownPos.y);
      if (dist > 6) return; // Ignore orbit drags
      const current = activeSceneRef.current;
      if (!current || current.interactables.length === 0) return;
      updatePointer(ev);
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(current.interactables, false);
      if (hits.length > 0) {
        current.onSelect(hits[0]);
      }
    };

    let lastHover = 0;
    const onPointerMove = (ev: PointerEvent) => {
      const now = performance.now();
      if (now - lastHover < 65) return;
      lastHover = now;
      const current = activeSceneRef.current;
      if (!current || current.interactables.length === 0) {
        renderer.domElement.style.cursor = current?.enableOrbit ? 'grab' : 'default';
        return;
      }
      updatePointer(ev);
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(current.interactables, false);
      renderer.domElement.style.cursor =
        hits.length > 0 ? 'pointer' : current.enableOrbit ? 'grab' : 'default';
    };

    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('pointerup', onPointerUp);
    renderer.domElement.addEventListener('pointermove', onPointerMove);

    let resizeRafId = 0;
    const resizeObserver = new ResizeObserver(() => {
      cancelAnimationFrame(resizeRafId);
      resizeRafId = requestAnimationFrame(() => {
        if (!container || disposed) return;
        const w = Math.max(240, container.clientWidth);
        const h = Math.max(240, container.clientHeight);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h, false);
        labelRenderer.setSize(w, h);
        renderer.render(scene, camera);
        labelRenderer.render(scene, camera);
      });
    });
    resizeObserver.observe(container);

    const startTime = performance.now();
    let prevTime = startTime;
    let frameId = 0;
    let disposed = false;

    const afterDraw = (cb: () => void) => {
      const gl = renderer.getContext() as WebGL2RenderingContext;
      const sync = gl.fenceSync ? gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0) : null;
      gl.flush();
      const poll = () => {
        if (disposed) return;
        const status = sync ? gl.getSyncParameter(sync, gl.SYNC_STATUS) : gl.SIGNALED;
        if (sync && status === gl.UNSIGNALED) {
          frameId = window.setTimeout(poll, 5);
        } else {
          if (sync) gl.deleteSync(sync);
          frameId = window.setTimeout(cb, 80);
        }
      };
      frameId = window.setTimeout(poll, 5);
    };

    const tick = () => {
      if (disposed) return;
      const now = performance.now();
      const delta = Math.min((now - prevTime) / 1000, 0.1);
      const elapsed = (now - startTime) / 1000;
      prevTime = now;

      if (controls.enabled) {
        controls.update();
      }
      activeSceneRef.current?.update(elapsed, delta, stateRef.current);
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);

      if (isSoftwareGl) {
        afterDraw(tick);
      } else {
        frameId = requestAnimationFrame(tick);
      }
    };

    // Build initial slide scene
    const initialBuilt = buildSlideScene(slideId, {
      onSelectFeature: (i) => callbacksRef.current.onSelectFeature?.(i),
      onSelectPillar: (i) => callbacksRef.current.onSelectPillar?.(i),
      onSelectMatrixCell: (r, c) => callbacksRef.current.onSelectMatrixCell?.(r, c),
      onSelectNeuron: (id) => callbacksRef.current.onSelectNeuron?.(id),
      onSelectOptimizer: (o) => callbacksRef.current.onSelectOptimizer?.(o),
      onSelectLoopStage: (s) => callbacksRef.current.onSelectLoopStage?.(s),
      onSelectStep: (s) => callbacksRef.current.onSelectStep?.(s),
      onSelectAuthor: (a) => callbacksRef.current.onSelectAuthor?.(a),
      onTriggerPulse: () => callbacksRef.current.onTriggerPulse?.(),
      onSnapSvm: () => callbacksRef.current.onSnapSvm?.(),
      onToggleFitMode: () => callbacksRef.current.onToggleFitMode?.(),
      onResampleData: () => callbacksRef.current.onResampleData?.(),
      onSelectMseSample: (idx) => callbacksRef.current.onSelectMseSample?.(idx),
    });
    activeSceneRef.current = initialBuilt;
    scene.add(initialBuilt.group);
    camera.position.copy(initialBuilt.cameraPos);
    controls.target.copy(initialBuilt.cameraTarget);
    controls.enabled = initialBuilt.enableOrbit;
    controls.update();

    if (!isSoftwareGl) {
      renderer.compile(scene, camera);
    }

    tick();

    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      cancelAnimationFrame(resizeRafId);
      clearTimeout(frameId);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('pointerup', onPointerUp);
      renderer.domElement.removeEventListener('pointermove', onPointerMove);
      renderer.domElement.removeEventListener('webglcontextlost', handleContextLost);
      renderer.domElement.removeEventListener('webglcontextrestored', handleContextRestored);
      activeSceneRef.current?.dispose();
      envRT.dispose();
      controls.dispose();
      renderer.dispose();
      labelRenderer.domElement.innerHTML = '';
      if (labelRenderer.domElement.parentNode === labelContainer) {
        labelContainer.removeChild(labelRenderer.domElement);
      }
      labelRendererRef.current = null;
      rendererRef.current = null;
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Swap 3D scene cleanly when slideId changes
  useEffect(() => {
    const scene = sceneRootRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const labelRenderer = labelRendererRef.current;
    if (!scene || !camera || !controls) return;

    if (activeSceneRef.current) {
      scene.remove(activeSceneRef.current.group);
      activeSceneRef.current.dispose();
    }
    if (labelRenderer) {
      labelRenderer.domElement.innerHTML = '';
    }

    const built = buildSlideScene(slideId, {
      onSelectFeature: (i) => callbacksRef.current.onSelectFeature?.(i),
      onSelectPillar: (i) => callbacksRef.current.onSelectPillar?.(i),
      onSelectMatrixCell: (r, c) => callbacksRef.current.onSelectMatrixCell?.(r, c),
      onSelectNeuron: (id) => callbacksRef.current.onSelectNeuron?.(id),
      onSelectOptimizer: (o) => callbacksRef.current.onSelectOptimizer?.(o),
      onSelectLoopStage: (s) => callbacksRef.current.onSelectLoopStage?.(s),
      onSelectStep: (s) => callbacksRef.current.onSelectStep?.(s),
      onSelectAuthor: (a) => callbacksRef.current.onSelectAuthor?.(a),
      onTriggerPulse: () => callbacksRef.current.onTriggerPulse?.(),
      onSnapSvm: () => callbacksRef.current.onSnapSvm?.(),
      onToggleFitMode: () => callbacksRef.current.onToggleFitMode?.(),
      onResampleData: () => callbacksRef.current.onResampleData?.(),
      onSelectMseSample: (idx) => callbacksRef.current.onSelectMseSample?.(idx),
    });

    activeSceneRef.current = built;
    scene.add(built.group);
    camera.position.copy(built.cameraPos);
    controls.target.copy(built.cameraTarget);
    controls.enabled = built.enableOrbit;
    controls.update();
    built.onStateChange(stateRef.current, camera, controls);
    rendererRef.current?.render(scene, camera);
    labelRenderer?.render(scene, camera);
  }, [slideId]);

  // Propagate interactive control state to active 3D scene
  useEffect(() => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const scene = sceneRootRef.current;
    if (!camera) return;
    activeSceneRef.current?.onStateChange(sceneState, camera, controls);
    if (scene) {
      rendererRef.current?.render(scene, camera);
      labelRendererRef.current?.render(scene, camera);
    }
  }, [sceneState]);

  return (
    <div className="relative w-full h-full min-h-[300px] border-2 border-[#14213d] bg-[#f4ede0] overflow-hidden">
      <div
        ref={mountRef}
        data-testid="three-canvas-container"
        data-slide-id={slideId}
        className="w-full h-full"
      />
      <div
        ref={labelMountRef}
        data-testid="three-label-overlay"
        className="pointer-events-none absolute inset-0 overflow-hidden"
      />
      {contextLost && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#f4ede0]/95 p-6 text-center">
          <div className="border-2 border-[#14213d] bg-white px-6 py-4">
            <p className="font-bold text-[#14213d]">3D Viewport Context Paused</p>
            <p className="text-sm text-[#14213d]/80 mt-1">
              Restoring WebGL rendering surface...
            </p>
          </div>
        </div>
      )}
      <div className="pointer-events-none absolute bottom-0 inset-x-0 flex items-center justify-between gap-3 border-t border-[#14213d]/20 bg-[#f4ede0]/95 px-3 py-1.5 text-xs text-[#14213d]">
        <span className="truncate font-medium">{interactionHint}</span>
        <span className="font-code shrink-0 text-[11px] font-semibold text-[#14213d]/75">
          CppTorch
        </span>
      </div>
    </div>
  );
}
