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

    // 3. Launch Aether Silicon Host (Backend AI Model)
    console.log("\n[2/3] Launching Aether Silicon Host in background...");
    const bootstrapperPath = path.join(buildDir, 'deployment', 'Release', 'Aether_AI_IDE.exe');
    
    if (!fs.existsSync(bootstrapperPath)) {
        throw new Error("Bootstrapper binary missing after build!");
    }

    const gatewayVbs = `CreateObject("WScript.Shell").Run "cmd /c cd ""${path.dirname(bootstrapperPath)}"" && ""${bootstrapperPath}""", 0, False`;
    fs.writeFileSync(path.join(buildDir, 'run_gateway.vbs'), gatewayVbs);
    execSync(`wscript "${path.join(buildDir, 'run_gateway.vbs')}"`);
    
    // 4. Launch VSCode Environment
    console.log("\n[3/3] Launching VSCode IDE Environment...");
    const vscodePath = path.join(__dirname, 'aether-agent');
    spawn('code', [vscodePath], {
        detached: true,
        stdio: 'ignore',
        shell: true
    }).unref();
    
    console.log("[SUCCESS] Aether Pipeline dispatched. VSCode IDE is now opening.\n");
    process.exit(0);

} catch (error) {
    console.error("\n[FATAL] Automated build/launch failed.");
    console.error(error.message);
    process.exit(1);
}
