'use client';

import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

interface NodeData {
  id: number;
  x: number;
  y: number;
  z: number;
  sequence: string;
  type: string;
  frequency: number;
}

interface EdgeData {
  source: number;
  target: number;
  frequency: number;
  attention: number;
  cohorts: string[];
}

interface Visualizer3DProps {
  nodes: NodeData[];
  edges: EdgeData[];
  selectedNodeId: number | null;
  onSelectNode: (nodeId: number | null) => void;
  selectedCohort: string;
  showAttention: boolean;
  onLogMessage: (msg: string, type?: 'success' | 'info' | 'warning') => void;
}

export default function Visualizer3D({
  nodes,
  edges,
  selectedNodeId,
  onSelectNode,
  selectedCohort,
  showAttention,
  onLogMessage
}: Visualizer3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const objectsGroupRef = useRef<THREE.Group | null>(null);
  const raycasterRef = useRef<THREE.Raycaster | null>(null);
  const mouseRef = useRef<THREE.Vector2>(new THREE.Vector2());

  // Store references to nodes to match raycasting
  const nodeMeshesRef = useRef<{ mesh: THREE.Mesh; id: number }[]>([]);

  useEffect(() => {
    if (!containerRef.current) return;

    onLogMessage('Initializing WebGL 3D Canvas context with biological DNA templates...', 'info');

    // 1. Scene Setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8fafc); // Light slate biological background
    scene.fog = new THREE.FogExp2(0xf8fafc, 0.012);
    sceneRef.current = scene;

    // 2. Camera Setup
    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    camera.position.set(0, 5, 55); // Elevated slightly for a nicer perspective
    cameraRef.current = camera;

    // 3. Renderer Setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x10b981, 0.9); // Emerald key light
    dirLight1.position.set(20, 40, 20);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x0ea5e9, 0.6); // Cyan fill light
    dirLight2.position.set(-20, -40, -20);
    scene.add(dirLight2);

    // 5. Objects Group
    const objectsGroup = new THREE.Group();
    scene.add(objectsGroup);
    objectsGroupRef.current = objectsGroup;

    // 6. Raycaster
    raycasterRef.current = new THREE.Raycaster();

    // 7. Mouse Interaction Handles
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };

    const handleMouseDown = (e: MouseEvent) => {
      isDragging = true;
      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        const deltaX = e.clientX - previousMousePosition.x;
        const deltaY = e.clientY - previousMousePosition.y;

        objectsGroup.rotation.y += deltaX * 0.005;
        objectsGroup.rotation.x += deltaY * 0.003; // dampened vertical rotation

        previousMousePosition = { x: e.clientX, y: e.clientY };
      }

      const rect = renderer.domElement.getBoundingClientRect();
      mouseRef.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    };

    const handleMouseUp = () => {
      isDragging = false;
    };

    const handleClick = () => {
      if (!raycasterRef.current || !cameraRef.current) return;

      raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);
      const meshes = nodeMeshesRef.current.map(item => item.mesh);
      const intersects = raycasterRef.current.intersectObjects(meshes);

      if (intersects.length > 0) {
        const hitMesh = intersects[0].object as THREE.Mesh;
        const hitNode = nodeMeshesRef.current.find(item => item.mesh === hitMesh);
        if (hitNode) {
          onSelectNode(hitNode.id);
          onLogMessage(`Selected allele segment ID ${hitNode.id} - Running sequence attention check...`, 'success');
        }
      } else {
        onSelectNode(null);
      }
    };

    const handleWheel = (e: WheelEvent) => {
      camera.position.z += e.deltaY * 0.04;
      camera.position.z = Math.max(15, Math.min(camera.position.z, 120));
    };

    const dom = renderer.domElement;
    dom.addEventListener('mousedown', handleMouseDown);
    dom.addEventListener('mousemove', handleMouseMove);
    dom.addEventListener('mouseup', handleMouseUp);
    dom.addEventListener('click', handleClick);
    dom.addEventListener('wheel', handleWheel);

    let animationFrameId: number;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      
      if (!isDragging) {
        objectsGroup.rotation.y += 0.001; // slow auto spin
      }
      
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!containerRef.current || !cameraRef.current || !rendererRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      dom.removeEventListener('mousedown', handleMouseDown);
      dom.removeEventListener('mousemove', handleMouseMove);
      dom.removeEventListener('mouseup', handleMouseUp);
      dom.removeEventListener('click', handleClick);
      dom.removeEventListener('wheel', handleWheel);
      if (dom && dom.parentNode) {
        dom.parentNode.removeChild(dom);
      }
      renderer.dispose();
    };
  }, []);

  // Update scene when nodes/edges change
  useEffect(() => {
    const scene = sceneRef.current;
    const group = objectsGroupRef.current;
    if (!scene || !group) return;

    // Clear previous objects
    while (group.children.length > 0) {
      const obj = group.children[0];
      group.remove(obj);
    }
    nodeMeshesRef.current = [];

    // --- BIOLOGICAL ADDITION: 3D DNA DOUBLE HELIX REFERENCE BACKBONE ---
    // This represents the canonical GRCh38 linear chromosome line.
    // The structural variation nodes and alternate paths will branch off and return to this backbone.
    const minX = -32;
    const maxX = 32;
    const steps = 120;
    const amplitude = 1.8;      // spiral width
    const frequency = 0.45;     // tightness of spiral
    
    const helixPoints1: THREE.Vector3[] = [];
    const helixPoints2: THREE.Vector3[] = [];

    for (let i = 0; i <= steps; i++) {
      const pct = i / steps;
      const x = minX + pct * (maxX - minX);
      const theta = x * frequency;

      // Generate helical spiral coordinates along X axis
      helixPoints1.push(new THREE.Vector3(x, amplitude * Math.sin(theta), amplitude * Math.cos(theta)));
      helixPoints2.push(new THREE.Vector3(x, -amplitude * Math.sin(theta), -amplitude * Math.cos(theta)));
    }

    // Draw Helix strand lines
    const helixCurve1 = new THREE.CatmullRomCurve3(helixPoints1);
    const helixGeom1 = new THREE.TubeGeometry(helixCurve1, 100, 0.12, 8, false);
    const helixMat1 = new THREE.MeshStandardMaterial({
      color: 0x94a3b8, // Light slate gray
      roughness: 0.2,
      metalness: 0.8,
      transparent: true,
      opacity: 0.35
    });
    const strand1 = new THREE.Mesh(helixGeom1, helixMat1);
    group.add(strand1);

    const helixCurve2 = new THREE.CatmullRomCurve3(helixPoints2);
    const helixGeom2 = new THREE.TubeGeometry(helixCurve2, 100, 0.12, 8, false);
    const strand2 = new THREE.Mesh(helixGeom2, helixMat1);
    group.add(strand2);

    // Draw base-pair rungs connecting the strands to reinforce DNA visual
    for (let i = 4; i < steps; i += 4) {
      const p1 = helixPoints1[i];
      const p2 = helixPoints2[i];
      
      const rungGeom = new THREE.CylinderGeometry(0.05, 0.05, p1.distanceTo(p2), 6);
      const rung = new THREE.Mesh(rungGeom, helixMat1);
      
      // Position and orient cylinder between points
      const position = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
      rung.position.copy(position);
      rung.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3().subVectors(p2, p1).normalize());
      group.add(rung);
    }
    // -------------------------------------------------------------------

    onLogMessage(`Rendering biological topological graph overlays...`, 'info');

    // Render Edges
    edges.forEach(edge => {
      const sourceNode = nodes.find(n => n.id === edge.source);
      const targetNode = nodes.find(n => n.id === edge.target);

      if (!sourceNode || !targetNode) return;

      const inCohort = selectedCohort === 'all' || edge.cohorts.includes(selectedCohort);
      if (!inCohort) return;

      let color = 0xcbd5e1; // Default soft slate
      let thickness = 0.08;
      let opacity = 0.55;

      if (showAttention) {
        if (edge.attention > 0.8) {
          color = 0xec4899; // Pathological path (Magenta glow)
          thickness = 0.3;
          opacity = 0.95;
        } else if (edge.attention > 0.5) {
          color = 0x0ea5e9; // Imputed transition (Sky Blue)
          thickness = 0.2;
          opacity = 0.8;
        } else {
          color = 0x6366f1; // Background connection
          thickness = 0.1;
          opacity = 0.5;
        }
      } else {
        if (edge.cohorts.includes('African') && selectedCohort === 'African') {
          color = 0x10b981; // Emerald
          thickness = 0.16;
          opacity = 0.8;
        } else if (edge.cohorts.includes('Ashkenazi') && selectedCohort === 'Ashkenazi') {
          color = 0xf59e0b; // Amber
          thickness = 0.16;
          opacity = 0.8;
        } else if (edge.cohorts.includes('East_Asian') && selectedCohort === 'East_Asian') {
          color = 0x8b5cf6; // Purple
          thickness = 0.16;
          opacity = 0.8;
        } else if (edge.cohorts.includes('European') && selectedCohort === 'European') {
          color = 0x0ea5e9; // Sky blue
          thickness = 0.16;
          opacity = 0.8;
        }
      }

      // Draw thick 3D Tubes instead of flat 2D lines for premium clinical mapping
      const start = new THREE.Vector3(sourceNode.x, sourceNode.y, sourceNode.z);
      const end = new THREE.Vector3(targetNode.x, targetNode.y, targetNode.z);
      
      const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
      const normal = new THREE.Vector3(
        -(end.y - start.y),
        (end.x - start.x),
        0
      ).normalize().multiplyScalar(3.5 * (edge.source % 2 === 0 ? 1 : -1));
      
      const controlPoint = new THREE.Vector3().addVectors(mid, normal);
      const curve = new THREE.QuadraticBezierCurve3(start, controlPoint, end);
      
      const tubeGeom = new THREE.TubeGeometry(curve, 20, thickness, 8, false);
      const tubeMat = new THREE.MeshStandardMaterial({
        color: color,
        roughness: 0.1,
        metalness: 0.5,
        transparent: true,
        opacity: opacity,
        blending: showAttention && edge.attention > 0.5 ? THREE.AdditiveBlending : THREE.NormalBlending
      });

      const tube = new THREE.Mesh(tubeGeom, tubeMat);
      group.add(tube);
    });

    // Render Nodes as sequence block capsules
    nodes.forEach(node => {
      const isSelected = node.id === selectedNodeId;
      
      // Node dimensions: thicker capsule if selected
      let radius = isSelected ? 1.1 : 0.7;
      let height = isSelected ? 2.5 : 1.8;
      
      let color = 0x64748b; // Slate reference base
      let emissive = 0x000000;

      if (node.type.includes('Insertion')) {
        color = 0x10b981; // Emerald block
        emissive = 0x064e3b;
      } else if (node.type.includes('Deletion')) {
        color = 0xef4444; // Rose red block
        emissive = 0x7f1d1d;
      } else if (node.type.includes('Polymorphic') || node.type.includes('Translocation')) {
        color = 0x8b5cf6; // Purple hypervariable block
        emissive = 0x4c1d95;
      } else {
        color = 0x0ea5e9; // Cyan reference block
        emissive = 0x0369a1;
      }

      if (isSelected) {
        color = 0xffffff;
        emissive = 0xec4899; // Magenta glow border selection
      }

      // Capsule geometry makes nodes look like segment sequence bars/nucleosomes
      const geometry = new THREE.CapsuleGeometry(radius, height, 4, 16);
      const material = new THREE.MeshStandardMaterial({
        color: color,
        emissive: emissive,
        emissiveIntensity: isSelected ? 2.5 : 0.35,
        roughness: 0.15,
        metalness: 0.8,
        transparent: true,
        opacity: 0.95
      });

      const capsule = new THREE.Mesh(geometry, material);
      capsule.position.set(node.x, node.y, node.z);
      
      // Rotate capsule slightly so they stand vertically
      capsule.rotation.z = Math.PI / 2;
      group.add(capsule);

      nodeMeshesRef.current.push({ mesh: capsule, id: node.id });
    });

  }, [nodes, edges, selectedCohort, showAttention, selectedNodeId]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      
      {/* Floating Biological Legend Overlay */}
      <div className="glass-panel" style={{
        position: 'absolute',
        bottom: '20px',
        left: '20px',
        maxWidth: '320px',
        padding: '12px',
        fontSize: '12px',
        pointerEvents: 'none',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        background: 'rgba(255, 255, 255, 0.85)',
        border: '1px solid #cbd5e1'
      }}>
        <div style={{ fontWeight: 'bold', textTransform: 'uppercase', fontSize: '10.5px', color: 'hsl(var(--text-secondary))', borderBottom: '1px solid #e2e8f0', paddingBottom: '4px' }}>
          🧬 Pangenome Graph Map Legend
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '16px', height: '6px', borderRadius: '3px', background: '#94a3b8', opacity: 0.4 }} />
          <span><strong>Linear Reference:</strong> GRCh38 DNA Double Helix backbone</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '12px', height: '12px', borderRadius: '3px', background: '#0ea5e9' }} />
          <span><strong>Reference Block:</strong> Conserved sequence segment</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '12px', height: '12px', borderRadius: '3px', background: '#10b981' }} />
          <span><strong>Insertion Block:</strong> Added sequence track (SV &ge; 50bp)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '12px', height: '12px', borderRadius: '3px', background: '#ef4444' }} />
          <span><strong>Deletion Block:</strong> Deleted sequence track (SV &ge; 50bp)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '12px', height: '12px', borderRadius: '3px', background: '#8b5cf6' }} />
          <span><strong>Polymorphic Block:</strong> Hypervariable segment</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', borderTop: '1px solid #e2e8f0', paddingTop: '4px', marginTop: '2px' }}>
          <div style={{ width: '16px', height: '4px', background: 'linear-gradient(90deg, #0ea5e9, #ec4899)' }} />
          <span><strong>Edge Weight:</strong> Thickness reflects GNN Attention ($\alpha$)</span>
        </div>
      </div>
    </div>
  );
}
