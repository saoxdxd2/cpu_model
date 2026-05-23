const { execSync, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

console.log("========================================================");
console.log(" NCA AETHER — AUTOMATED SILICON BUILD & LAUNCH");
console.log("========================================================\n");

const buildDir = path.join(__dirname, 'build');

try {
    // 1. Ensure Build Directory exists
    if (!fs.existsSync(buildDir)) {
        console.log("[INIT] Creating build environment...");
        execSync('mkdir build', { stdio: 'inherit' });
        execSync('cd build && cmake .. -DCMAKE_PREFIX_PATH="C:\\libtorch"', { stdio: 'inherit' });
    }

    // 2. Synchronize Hardware Kernels (Auto-Compile)
    console.log("[1/2] Synchronizing Hardware Kernels (Auto-Compile)...");
    execSync('cmake --build build --config Release', { stdio: 'inherit' });
    console.log("\n[OK] Build Saturated.");

    // 3. Launch Aether AI IDE Ground
    console.log("\n[2/2] Launching Aether AI IDE Ground...");
    const bootstrapperPath = path.join(buildDir, 'deployment', 'Release', 'Aether_AI_IDE.exe');
    
    if (!fs.existsSync(bootstrapperPath)) {
        throw new Error("Bootstrapper binary missing after build!");
    }

    // Spawn the bootstrapper as a detached process
    const child = spawn(bootstrapperPath, [], {
        detached: true,
        stdio: 'inherit',
        cwd: path.dirname(bootstrapperPath)
    });
    
    child.unref();
    console.log("[SUCCESS] Aether Pipeline dispatched. Node process exiting.\n");
    process.exit(0);

} catch (error) {
    console.error("\n[FATAL] Automated build/launch failed.");
    console.error(error.message);
    process.exit(1);
}
