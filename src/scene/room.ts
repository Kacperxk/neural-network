/** The room the network stands in: floor, back wall, bench and lights. All geometry is procedural. */
import {
  AdditiveBlending,
  BoxGeometry,
  CanvasTexture,
  Color,
  DirectionalLight,
  Group,
  HemisphereLight,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  PlaneGeometry,
  PointLight,
} from 'three';
import { SIZES } from '../engine/params';
import { BENCH_TOP, COL_X, COLORS, neuronPos } from './theme';

export type Room = { group: Group; lights: Group };

/** A soft radial glow, drawn into a canvas: used as fake bounce light on the wall and floor. */
function glowTexture(inner: string, outer: string): CanvasTexture {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d')!;
  const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  grad.addColorStop(0, inner);
  grad.addColorStop(0.45, outer);
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 256, 256);
  return new CanvasTexture(c);
}

export function buildRoom(): Room {
  const group = new Group();
  const lights = new Group();

  const floor = new Mesh(
    new PlaneGeometry(90, 90),
    new MeshStandardMaterial({ color: 0x131b2e, roughness: 0.38, metalness: 0.5 }),
  );
  floor.rotation.x = -Math.PI / 2;
  group.add(floor);

  const wallMat = new MeshStandardMaterial({ color: 0x141c30, roughness: 0.9, metalness: 0.1 });
  const wall = new Mesh(new PlaneGeometry(60, 20), wallMat);
  wall.position.set(0, 10, -8);
  group.add(wall);

  // Wall panels and warm light slots: gives the back of the room some depth without competing with the network.
  const panelMat = new MeshStandardMaterial({ color: 0x1a2440, roughness: 0.75, metalness: 0.25 });
  const warm = new MeshBasicMaterial({ color: new Color(0xffb86b).multiplyScalar(1.3) });
  const cool = new MeshBasicMaterial({ color: new Color(COLORS.fwd).multiplyScalar(0.9) });
  const slotWarm = new MeshBasicMaterial({ color: new Color(0xffb86b).multiplyScalar(0.55) });
  const slotCool = new MeshBasicMaterial({ color: new Color(COLORS.fwd).multiplyScalar(0.4) });
  for (let i = -3; i <= 3; i++) {
    const panel = new Mesh(new BoxGeometry(4.9, 9.5, 0.2), panelMat);
    panel.position.set(i * 5.2, 5.2, -7.85);
    group.add(panel);
  }
  // Light slots sit outside the network's silhouette so they never read as part of it.
  for (const [x, mat] of [
    [-13, slotCool],
    [-7.8, slotWarm],
    [9.4, slotWarm],
    [14.6, slotCool],
  ] as const) {
    const slot = new Mesh(new BoxGeometry(0.06, 9.2, 0.06), mat);
    slot.position.set(x, 5.2, -7.7);
    group.add(slot);
  }
  const glowMat = (inner: string, outer: string, opacity: number): MeshBasicMaterial =>
    new MeshBasicMaterial({ map: glowTexture(inner, outer), transparent: true, opacity, blending: AdditiveBlending, depthWrite: false, fog: false });
  const wallGlow = new Mesh(new PlaneGeometry(26, 14), glowMat('rgba(255,170,95,0.55)', 'rgba(90,70,120,0.18)', 0.5));
  wallGlow.position.set(0.6, 4.2, -7.6);
  group.add(wallGlow);
  const floorGlow = new Mesh(new PlaneGeometry(24, 13), glowMat('rgba(255,160,80,0.6)', 'rgba(60,90,160,0.15)', 0.42));
  floorGlow.rotation.x = -Math.PI / 2;
  floorGlow.position.set(0.6, 0.012, 2.6);
  group.add(floorGlow);

  // Bench.
  const benchMat = new MeshStandardMaterial({ color: 0x232e48, roughness: 0.5, metalness: 0.55 });
  const bench = new Mesh(new BoxGeometry(13.6, BENCH_TOP - 0.08, 4.4), benchMat);
  bench.position.set(0.6, (BENCH_TOP - 0.08) / 2 + 0.08, 0);
  group.add(bench);
  const plinth = new Mesh(new BoxGeometry(13.0, 0.08, 3.9), new MeshStandardMaterial({ color: 0x090c14, roughness: 0.8 }));
  plinth.position.set(0.6, 0.04, 0);
  group.add(plinth);
  const top = new Mesh(new BoxGeometry(13.3, 0.03, 4.1), new MeshStandardMaterial({ color: 0x2c3957, roughness: 0.32, metalness: 0.6 }));
  top.position.set(0.6, BENCH_TOP + 0.005, 0);
  group.add(top);
  const strip = new Mesh(new BoxGeometry(13.62, 0.035, 0.02), warm);
  strip.position.set(0.6, BENCH_TOP - 0.1, 2.205);
  group.add(strip);

  // One slim stand behind each column, so the network reads as apparatus rather than floating dots.
  const standMat = new MeshStandardMaterial({ color: 0x34405e, roughness: 0.4, metalness: 0.7 });
  for (let col = 0; col < SIZES.length; col++) {
    const topY = neuronPos(col, 0).y + 0.25;
    const stand = new Mesh(new BoxGeometry(0.07, topY - BENCH_TOP, 0.07), standMat);
    stand.position.set(COL_X[col], (topY + BENCH_TOP) / 2, -0.62);
    group.add(stand);
    const foot = new Mesh(new BoxGeometry(0.7, 0.06, 0.5), standMat);
    foot.position.set(COL_X[col], BENCH_TOP + 0.03, -0.62);
    group.add(foot);
    for (let i = 0; i < SIZES[col]; i++) {
      const arm = new Mesh(new BoxGeometry(0.04, 0.04, 0.3), standMat);
      const p = neuronPos(col, i);
      arm.position.set(p.x, p.y, -0.5);
      group.add(arm);
    }
    const lamp = new Mesh(new BoxGeometry(0.34, 0.025, 0.02), col === 0 || col === 3 ? cool : warm);
    lamp.position.set(COL_X[col], BENCH_TOP + 0.075, -0.36);
    group.add(lamp);
  }

  lights.add(new HemisphereLight(0x9db8ff, 0x1a1420, 1.0));
  const key = new DirectionalLight(0xffd9b0, 2.2);
  key.position.set(6, 10, 7);
  lights.add(key);
  const rim = new DirectionalLight(0x6fbfff, 0.7);
  rim.position.set(-8, 5, -4);
  lights.add(rim);
  const glow = new PointLight(0xffb070, 30, 18, 2);
  glow.position.set(0, 1.6, 3.5);
  lights.add(glow);

  return { group, lights };
}
