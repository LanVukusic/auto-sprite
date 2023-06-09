import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import jsonData from "./embedding.json";

const data = jsonData as any as {
  embedding: number[];
  name: string;
}[];

// Create a scene
const scene = new THREE.Scene();
scene.background = new THREE.Color("white");

// Create a camera
const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.01,
  3000
);
camera.position.z = -3;
camera.position.x = 20;

// Create a renderer
const renderer = new THREE.WebGLRenderer({
  antialias: false,
});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
// renderer.logarithmicDepthBuffer = true;
document.body.appendChild(renderer.domElement);

// Create an instance of OrbitControls
const controls = new OrbitControls(camera, renderer.domElement);

// Create a sprite
const textureLoader = new THREE.TextureLoader();

data.forEach((d) => {
  const texture = textureLoader.load("/swords/" + d.name);
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;

  const DIVISION = 0.5;
  // const texture = textureLoader.load("/swords/" + d.image);
  const spriteMaterial = new THREE.SpriteMaterial({ map: texture, fog: false });
  const sprite = new THREE.Sprite(spriteMaterial);
  sprite.position.set(
    // Math.log(d.embedding[0] ** 2 / DIVISION),
    // Math.log(d.embedding[1] ** 2 / DIVISION),
    // Math.log(d.embedding[2] ** 2 / DIVISION)
    d.embedding[0] / DIVISION,
    d.embedding[1] / DIVISION,
    d.embedding[2] / DIVISION
  );
  const s = 3;
  sprite.scale.set(s, s, s);
  scene.add(sprite);
});

// const texture = textureLoader.load("/swords/3uo1s690gcb31_2.png");
// const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
// const sprite = new THREE.Sprite(spriteMaterial);
// sprite.scale.set(0.2, 0.2, 0.2);
// scene.add(sprite);

// Animation loop
function animate() {
  requestAnimationFrame(animate);

  controls.update();
  renderer.render(scene, camera);
}

animate();
