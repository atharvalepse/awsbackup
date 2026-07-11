"use client";

/**
 * Hero decoration — a small, precise preview of the memory graph.
 * ~150 instanced spheres in 6 clusters, nearest-neighbour lines, slow orbit,
 * subtle pointer parallax, one bloom pass. Capped DPR, pauses when tab hidden.
 *
 * Loaded via next/dynamic({ssr:false}) so three.js never ships in the LCP path.
 */
import { useLayoutEffect, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import { useReducedMotion } from "framer-motion";
import * as THREE from "three";
import { CLUSTER_HEX } from "@/lib/cluster-colors";

const COUNT = 150;
const CLUSTERS = 6;

function useGraphGeometry() {
  return useMemo(() => {
    // Cluster centers spread over a sphere (golden-angle spiral).
    const centers = Array.from({ length: CLUSTERS }, (_, i) => {
      const phi = Math.acos(1 - (2 * (i + 0.5)) / CLUSTERS);
      const theta = i * 2.399963;
      return new THREE.Vector3().setFromSphericalCoords(2.5, phi, theta);
    });

    const pts: { p: THREE.Vector3; c: number }[] = [];
    for (let i = 0; i < COUNT; i++) {
      const c = i % CLUSTERS;
      const jitter = new THREE.Vector3(
        Math.random() - 0.5,
        Math.random() - 0.5,
        Math.random() - 0.5,
      ).multiplyScalar(1.05);
      pts.push({ p: centers[c].clone().add(jitter), c });
    }

    // One line per point → its nearest same-cluster neighbour.
    const linePos: number[] = [];
    for (let i = 0; i < pts.length; i++) {
      let best = -1;
      let bestD = Infinity;
      for (let j = 0; j < pts.length; j++) {
        if (i === j || pts[i].c !== pts[j].c) continue;
        const d = pts[i].p.distanceToSquared(pts[j].p);
        if (d < bestD) {
          bestD = d;
          best = j;
        }
      }
      if (best >= 0) {
        linePos.push(pts[i].p.x, pts[i].p.y, pts[i].p.z);
        linePos.push(pts[best].p.x, pts[best].p.y, pts[best].p.z);
      }
    }
    return { pts, linePositions: new Float32Array(linePos) };
  }, []);
}

function Graph({ reduce }: { reduce: boolean }) {
  const { pts, linePositions } = useGraphGeometry();
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const groupRef = useRef<THREE.Group>(null);
  const pointer = useRef({ x: 0, y: 0 });

  useLayoutEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    const color = new THREE.Color();
    pts.forEach((pt, i) => {
      dummy.position.copy(pt.p);
      const s = 0.06 + Math.random() * 0.07;
      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      mesh.setColorAt(i, color.set(CLUSTER_HEX[pt.c]));
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [pts]);

  useFrame(() => {
    if (reduce) return; // no parallax under reduced motion
    if (document.visibilityState === "hidden") return; // pause when tab hidden
    const g = groupRef.current;
    if (!g) return;
    // parallax tilt, max ±3° (~0.052 rad), eased toward pointer
    g.rotation.x += (pointer.current.y * 0.052 - g.rotation.x) * 0.05;
    g.rotation.y += (pointer.current.x * 0.052 - g.rotation.y) * 0.05;
  });

  return (
    <group
      ref={groupRef}
      onPointerMove={(e) => {
        pointer.current = {
          x: (e.pointer?.x ?? 0),
          y: (e.pointer?.y ?? 0),
        };
      }}
    >
      <instancedMesh ref={meshRef} args={[undefined, undefined, COUNT]}>
        <sphereGeometry args={[1, 12, 12]} />
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[linePositions, 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#8a8a96" transparent opacity={0.15} toneMapped={false} />
      </lineSegments>
    </group>
  );
}

export default function HeroScene() {
  // Respect prefers-reduced-motion: a static frame, no orbit, no continuous loop.
  const reduce = useReducedMotion() ?? false;
  return (
    <Canvas
      frameloop={reduce ? "demand" : "always"}
      dpr={[1, 1.5]}
      camera={{ position: [0, 0, 8], fov: 45 }}
      gl={{ antialias: true, alpha: true }}
      style={{ background: "transparent" }}
    >
      <Graph reduce={reduce} />
      <OrbitControls
        autoRotate={!reduce}
        autoRotateSpeed={0.3}
        enableZoom={false}
        enablePan={false}
        enableRotate={false}
      />
      <EffectComposer>
        <Bloom intensity={0.4} luminanceThreshold={0.85} luminanceSmoothing={0.2} mipmapBlur />
      </EffectComposer>
    </Canvas>
  );
}
