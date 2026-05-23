/**
 * NCA — Freedom Refactor Proof
 * Demonstrates the SiliconAutomation engine stripping telemetry
 * and limitations from the VSCode ground.
 */

#include "nn/core/execution/silicon_automation.hpp"
#include <iostream>
#include <vector>

using namespace nca::automation;

int main() {
    std::cout << "========================================================\n";
    std::cout << " NCA — FREEDOM REFACTOR (Silicon Automation)          \n";
    std::cout << "========================================================\n\n";

    // 1. Target the 'Ground' (VSCode Codebase)
    SiliconAutomation automation("vscode");

    // 2. Define the 'Big Company' cleanup list
    std::vector<std::string> freedom_targets = {
        "telemetry.publicLog",
        "authenticationService.signIn",
        "microsoft.com/telemetry",
        "Sign in to synchronize",
        "Enable telemetry"
    };

    // 3. Execute Silicon Wipe
    std::cout << "[1/2] Initiating Global Telemetry Wipe...\n";
    automation.silicon_wipe(freedom_targets);

    // 4. Automated Re-Branding
    std::cout << "[2/2] Injecting Aether Identity into IDE...\n";
    automation.batch_refactor("VS Code", "NCA AETHER IDE");
    automation.batch_refactor("Visual Studio Code", "Aether Pipeline");

    std::cout << "\n[SUCCESS] FREEDOM REFACTOR COMPLETE.\n";
    std::cout << "          The IDE is now an unrestricted Aether ground.\n";

    return 0;
}
