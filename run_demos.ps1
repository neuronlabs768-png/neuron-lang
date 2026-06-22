#!/usr/bin/env pwsh
# NEURON Demo Suite - Run this to see NEURON's three killer features

$neuronc = "$PSScriptRoot\target\release\neuronc.exe"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  NEURON -- The AI-Native Programming Language" -ForegroundColor Cyan  
Write-Host "  Three features no other language has." -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# Demo 1: The Million-Dollar Bug
Write-Host ""
Write-Host ""
Write-Host "  DEMO 1: The Million-Dollar Bug" -ForegroundColor Red
Write-Host "  Can your language catch lookahead bias at compile time?" -ForegroundColor Red
Write-Host "  -------------------------------------------------------" -ForegroundColor Red
Write-Host ""
Write-Host "  In Python/PyTorch, this bug silently passes:" -ForegroundColor White
Write-Host '    signal = model(price[t+1])   # accidental future leak' -ForegroundColor DarkGray
Write-Host '    # Trains fine. Backtests beautifully. Loses millions live.' -ForegroundColor DarkGray
Write-Host ""
Write-Host "  In NEURON:" -ForegroundColor Yellow
Write-Host ""

& $neuronc check "$PSScriptRoot\examples\demo_million_dollar_bug.nr" 2>&1 | ForEach-Object {
    $line = $_.ToString()
    if ($line -match "error") { Write-Host "    $line" -ForegroundColor Red }
    elseif ($line -match "help:") { Write-Host "    $line" -ForegroundColor Green }
    elseif ($line -match "expected:|got:") { Write-Host "    $line" -ForegroundColor Yellow }
    elseif ($line.Trim().Length -gt 0) { Write-Host "    $line" }
}

Write-Host ""
Write-Host "  >> NEURON caught at compile time what would have cost" -ForegroundColor Green
Write-Host "  >> millions in production. The compiler said no." -ForegroundColor Green

# Demo 2: Correlation != Causation
Write-Host ""
Write-Host ""
Write-Host "  DEMO 2: Correlation Is Not Causation" -ForegroundColor Magenta
Write-Host "  Can your language distinguish observation from intervention?" -ForegroundColor Magenta
Write-Host "  -------------------------------------------------------" -ForegroundColor Magenta
Write-Host ""
Write-Host "  Question: Will giving Treatment X improve outcomes?" -ForegroundColor White
Write-Host "  Traditional ML gives correlations. NEURON gives causation." -ForegroundColor White
Write-Host ""

& $neuronc check "$PSScriptRoot\examples\demo_causal.nr" 2>&1 | ForEach-Object {
    $line = $_.ToString()
    if ($line -match "error") { Write-Host "    $line" -ForegroundColor Red }
    elseif ($line -match "help:") { Write-Host "    $line" -ForegroundColor Green }
    elseif ($line.Trim().Length -gt 0) { Write-Host "    $line" }
}

Write-Host ""
Write-Host "  >> NEURON doesn't just predict what happens." -ForegroundColor Green
Write-Host "  >> It reasons about what WOULD happen if you changed the world." -ForegroundColor Green

# Demo 3: Machine Unlearning
Write-Host ""
Write-Host ""
Write-Host "  DEMO 3: Machine Unlearning" -ForegroundColor Blue
Write-Host "  Can your language forget on command and prove it?" -ForegroundColor Blue
Write-Host "  -------------------------------------------------------" -ForegroundColor Blue
Write-Host ""
Write-Host "  A hospital must delete patient data under GDPR Article 17." -ForegroundColor White
Write-Host "  The model must forget and produce a verifiable certificate." -ForegroundColor White
Write-Host ""
Write-Host "  > neuronc run examples/demo_forget.nr" -ForegroundColor Yellow
Write-Host ""

& $neuronc run "$PSScriptRoot\examples\demo_forget.nr" 2>&1 | ForEach-Object {
    $line = $_.ToString()
    if ($line.Trim().Length -gt 0) { Write-Host "    $line" -ForegroundColor Cyan }
}

Write-Host ""
Write-Host "  >> Machine unlearning is built into the language." -ForegroundColor Green
Write-Host "  >> ForgetCertificate with provable residual bounds." -ForegroundColor Green

# Summary
Write-Host ""
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  NEURON -- Nine structural guarantees." -ForegroundColor Cyan
Write-Host "  Tensor shapes. Gradients. Temporal safety. Causality." -ForegroundColor Cyan
Write-Host "  Uncertainty. Effects. Explainability. Merging. Unlearning." -ForegroundColor Cyan
Write-Host ""
Write-Host "  Built from Marsabit, Kenya." -ForegroundColor Yellow
Write-Host "  Designed to change what AI can be trusted to do." -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
