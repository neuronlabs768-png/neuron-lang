// NEURON Website Simulation Logic

document.addEventListener("DOMContentLoaded", () => {
  // 1. Code Snippets with Pre-Highlighted Syntax
  const codeSnippets = {
    temporal: `fn predict_price(prices: <span class="hl-type">Temporal</span>[Tensor, past_to_future]) -> <span class="hl-type">Tensor</span>:
    <span class="hl-keyword">let</span> prev_price = prices.before(<span class="hl-number">1</span>) <span class="hl-comment"># OK: reading historical price</span>
    <span class="hl-keyword">let</span> future_leak = prices.after(<span class="hl-number">2</span>)  <span class="hl-comment"># COMPILE ERROR: lookahead bias!</span>
    <span class="hl-keyword">return</span> future_leak`,

    causal: `<span class="hl-keyword">fn</span> treatment_analysis(patient_data: <span class="hl-type">Dataset</span>):
    <span class="hl-keyword">let</span> obs = observe(patient_data, treatment=<span class="hl-number">1</span>)  <span class="hl-comment"># P(Y|X)</span>
    <span class="hl-keyword">let</span> intervened = intervene(treatment=<span class="hl-number">1</span>)      <span class="hl-comment"># P(Y|do(X))</span>
    
    <span class="hl-comment"># TYPE ERROR: Cannot mix conditional observations with interventions</span>
    <span class="hl-keyword">let</span> effect: <span class="hl-type">Causal</span>[Intervention] = obs`,

    uncertainty: `<span class="hl-keyword">fn</span> autonomous_driving(lidar: <span class="hl-type">Uncertain</span>[Tensor[1, 3]]) -> <span class="hl-type">Tensor</span>:
    <span class="hl-keyword">let</span> distance = preprocess(lidar)
    
    <span class="hl-comment"># COMPILER WARNING: returning Uncertain prediction without check</span>
    <span class="hl-keyword">return</span> distance`,

    autograd: `<span class="hl-decorator">@differentiable</span>
<span class="hl-keyword">model</span> LinearNet:
  w: <span class="hl-type">Tensor</span>[4, 1] = glorot(<span class="hl-number">4</span>, <span class="hl-number">1</span>)

  <span class="hl-keyword">fn</span> train(self, x: <span class="hl-type">Tensor</span>[B, 4], y: <span class="hl-type">Tensor</span>[B, 1]) [<span class="hl-type">Effect</span>[Mut[self]]]:
    <span class="hl-keyword">let</span> loss = mse(x @ self.w, y)
    <span class="hl-keyword">update</span> self.w <span class="hl-keyword">by</span> sgd(grad(loss), lr=<span class="hl-number">0.1</span>)`
  };

  // 1b. Raw Code Snippets for Copy-to-Clipboard
  const rawSnippets = {
    temporal: `fn predict_price(prices: Temporal[Tensor, past_to_future]) -> Tensor:
    let prev_price = prices.before(1) # OK: reading historical price
    let future_leak = prices.after(2)  # COMPILE ERROR: lookahead bias!
    return future_leak`,

    causal: `fn treatment_analysis(patient_data: Dataset):
    let obs = observe(patient_data, treatment=1)  # P(Y|X)
    let intervened = intervene(treatment=1)      # P(Y|do(X))
    
    # TYPE ERROR: Cannot mix conditional observations with interventions
    let effect: Causal[Intervention] = obs`,

    uncertainty: `fn autonomous_driving(lidar: Uncertain[Tensor[1, 3]]) -> Tensor:
    let distance = preprocess(lidar)
    
    # COMPILER WARNING: returning Uncertain prediction without check
    return distance`,

    autograd: `@differentiable
model LinearNet:
  w: Tensor[4, 1] = glorot(4, 1)

  fn train(self, x: Tensor[B, 4], y: Tensor[B, 1]) [Effect[Mut[self]]]:
    let loss = mse(x @ self.w, y)
    update self.w by sgd(grad(loss), lr=0.1)`
  };

  // 2. Line counts for each snippet
  const lineCounts = {
    temporal: 4,
    causal: 6,
    uncertainty: 5,
    autograd: 8
  };

  // 3. Simulated Compiler Outputs
  const compilerLogs = {
    temporal: [
      { text: "visitor@neuron:~$ neuronc check examples/temporal_leak.nr", type: "prompt" },
      { text: "Analyzing temporal data dependencies...", type: "info" },
      { text: "[ERROR] Line 3: TemporalLeak detected.", type: "error" },
      { text: "  --> examples/temporal_leak.nr:3:21", type: "info" },
      { text: "   |", type: "info" },
      { text: " 3 |     let future_leak = prices.after(2)", type: "info" },
      { text: "   |                       ^^^^^^^^^^^^^^^ Lookahead violation: reading future timestamps.", type: "error" },
      { text: "   |", type: "info" },
      { text: "Compilation failed: 1 temporal type violation found.", type: "error" }
    ],
    causal: [
      { text: "visitor@neuron:~$ neuronc check examples/causal_engine.nr", type: "prompt" },
      { text: "Type-checking structural causal model variables...", type: "info" },
      { text: "[ERROR] Line 6: CausalTypeMismatch", type: "error" },
      { text: "  --> examples/causal_engine.nr:6:40", type: "info" },
      { text: "   |", type: "info" },
      { text: " 6 |     let effect: Causal[Intervention] = obs", type: "info" },
      { text: "   |                                        ^^^ expected Causal[Intervention], found Causal[Observation]", type: "error" },
      { text: "   |", type: "info" },
      { text: "Compilation failed: 1 causal type violation found.", type: "error" }
    ],
    uncertainty: [
      { text: "visitor@neuron:~$ neuronc check examples/lidar_test.nr", type: "prompt" },
      { text: "Analyzing uncertainty propagation pathways...", type: "info" },
      { text: "[WARNING] Line 5: UncheckedUncertainty", type: "warning" },
      { text: "  --> examples/lidar_test.nr:5:12", type: "info" },
      { text: "   |", type: "info" },
      { text: " 5 |     return distance", type: "info" },
      { text: "   |            ^^^^^^^^ returning Uncertain value without explicit confidence threshold check.", type: "warning" },
      { text: "   |", type: "info" },
      { text: "Compilation succeeded with 1 warning.", type: "success" }
    ],
    autograd: [
      { text: "visitor@neuron:~$ neuronc run examples/linear_regression.nr", type: "prompt" },
      { text: "Initializing AD tape & allocating tensors on JIT backend...", type: "info" },
      { text: "Compilation succeeded. Running JIT interpreter...", type: "success" },
      { text: "Iter 000/100: Loss = 16.000 (starting weight = 5.0)", type: "info" },
      { text: "Iter 020/100: Loss = 5.7600", type: "info" },
      { text: "Iter 040/100: Loss = 2.0736", type: "info" },
      { text: "Iter 060/100: Loss = 0.7464", type: "info" },
      { text: "Iter 080/100: Loss = 0.2687", type: "info" },
      { text: "Iter 100/100: Loss = 0.0001 (weight converged to 3.0)", type: "success" },
      { text: "Execution complete. Tape reset, 0 memory leaks.", type: "success" }
    ]
  };

  // 4. State Management
  let currentTab = "temporal";
  let isRunning = false;

  // 5. DOM Elements
  const tabContainer = document.getElementById("editor-tabs");
  const codeContainer = document.getElementById("code-container");
  const lineNumbersContainer = document.getElementById("line-numbers");
  const terminalBody = document.getElementById("terminal-body");
  const runBtn = document.getElementById("run-btn");
  const copyBtn = document.getElementById("copy-btn");
  const navToggle = document.getElementById("nav-toggle");
  const navLinks = document.querySelector(".nav-links");

  // 6. Initialize UI
  function initPlayground() {
    updateEditor();
  }

  function updateEditor() {
    // Set code content
    codeContainer.innerHTML = `<div class="code-block active">${codeSnippets[currentTab]}</div>`;
    
    // Set line numbers
    let lineHtml = "";
    for (let i = 1; i <= lineCounts[currentTab]; i++) {
      lineHtml += `${i}<br>`;
    }
    lineNumbersContainer.innerHTML = lineHtml;
  }

  // 7. Tab Switching Event Listeners
  tabContainer.addEventListener("click", (e) => {
    if (isRunning) return; // Prevent tab switching while compiling
    
    const button = e.target.closest(".tab-btn");
    if (!button) return;
    
    // Update active tab button style
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    button.classList.add("active");
    
    // Set tab state and update editor
    currentTab = button.dataset.tab;
    updateEditor();
    
    // Reset terminal
    terminalBody.innerHTML = `
      <div class="term-line"><span class="term-prompt">visitor@neuron:~$</span> neuronc check examples/${currentTab === 'autograd' ? 'linear_regression' : currentTab + '_leak'}.nr</div>
      <div class="term-line">Ready to compile. Click "Compile & Run" above to execute.</div>
    `;
  });

  // 8. Copy Snippet Event Listener
  copyBtn.addEventListener("click", () => {
    const rawText = rawSnippets[currentTab];
    navigator.clipboard.writeText(rawText).then(() => {
      const span = copyBtn.querySelector("span");
      span.textContent = "Copied!";
      setTimeout(() => {
        span.textContent = "Copy";
      }, 2000);
    }).catch(err => {
      console.error("Clipboard copy failed: ", err);
    });
  });

  // 9. Run Simulation
  runBtn.addEventListener("click", () => {
    if (isRunning) return;
    
    isRunning = true;
    runBtn.style.opacity = "0.6";
    runBtn.innerHTML = `
      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" style="animation: spin 1s linear infinite"><circle cx="12" cy="12" r="10"></circle><path d="M12 2v4"></path></svg>
      Compiling...
    `;
    
    // Clear terminal and prepare to type logs
    terminalBody.innerHTML = "";
    let logQueue = compilerLogs[currentTab];
    let logIndex = 0;
    
    function printNextLine() {
      if (logIndex >= logQueue.length) {
        // Finished simulation
        isRunning = false;
        runBtn.style.opacity = "1";
        runBtn.innerHTML = `
          <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
          Compile & Run
        `;
        return;
      }
      
      const log = logQueue[logIndex];
      const div = document.createElement("div");
      div.className = "term-line";
      
      if (log.type === "prompt") {
        div.innerHTML = `<span class="term-prompt">visitor@neuron:~$</span> ${log.text.replace("visitor@neuron:~$ ", "")}`;
      } else if (log.type === "error") {
        div.innerHTML = `<span class="term-error">${log.text}</span>`;
      } else if (log.type === "warning") {
        div.innerHTML = `<span class="term-warning">${log.text}</span>`;
      } else if (log.type === "success") {
        div.innerHTML = `<span class="term-success">${log.text}</span>`;
      } else {
        div.textContent = log.text;
      }
      
      terminalBody.appendChild(div);
      terminalBody.scrollTop = terminalBody.scrollHeight;
      
      logIndex++;
      
      // Add simulated delays for visual feedback
      let delay = 120;
      if (log.type === "prompt") delay = 300;
      if (log.text.includes("Executing") || log.text.includes("allocating")) delay = 500;
      if (log.text.includes("Iter 000")) delay = 400;
      // Use exact iteration number matching to prevent override order bug
      if (log.text.includes("Iter 0") && !log.text.includes("Iter 000")) delay = 100;
      
      setTimeout(printNextLine, delay);
    }
    
    printNextLine();
  });

  // 10. Mobile Navbar Toggle
  if (navToggle) {
    navToggle.addEventListener("click", () => {
      navLinks.classList.toggle("open");
      navToggle.classList.toggle("active");
    });
  }

  // 11. FAQ Accordion Event Listeners
  const faqItems = document.querySelectorAll(".faq-item");
  faqItems.forEach(item => {
    const question = item.querySelector(".faq-question");
    question.addEventListener("click", () => {
      const isActive = item.classList.contains("active");
      
      // Close other items
      faqItems.forEach(el => el.classList.remove("active"));
      
      // Toggle clicked item
      if (!isActive) {
        item.classList.add("active");
      }
    });
  });

  // 12. SaaS Pricing Billing Toggle Switch (Removed in public launch)

  // 13. Waitlist Modal Handling
  const waitlistModal = document.getElementById("waitlist-modal");
  const openWaitlistBtn = document.getElementById("open-waitlist-btn");
  const closeWaitlistBtn = document.getElementById("close-waitlist-btn");
  const successCloseBtn = document.getElementById("success-close-btn");
  const waitlistForm = document.getElementById("waitlist-form");
  const waitlistEmailInput = document.getElementById("waitlist-email");
  const waitlistSubmitBtn = document.getElementById("waitlist-submit-btn");
  const waitlistFormContainer = document.getElementById("waitlist-form-container");
  const waitlistSuccessContainer = document.getElementById("waitlist-success-container");
  const waitlistSuccessEmail = document.getElementById("waitlist-success-email");

  function openModal() {
    if (waitlistModal) {
      waitlistFormContainer.style.display = "block";
      waitlistSuccessContainer.style.display = "none";
      waitlistEmailInput.value = "";
      waitlistModal.classList.add("active");
    }
  }

  function closeModal() {
    if (waitlistModal) {
      waitlistModal.classList.remove("active");
    }
  }

  if (openWaitlistBtn) {
    openWaitlistBtn.addEventListener("click", openModal);
  }

  if (closeWaitlistBtn) {
    closeWaitlistBtn.addEventListener("click", closeModal);
  }

  if (successCloseBtn) {
    successCloseBtn.addEventListener("click", closeModal);
  }

  if (waitlistModal) {
    waitlistModal.addEventListener("click", (e) => {
      if (e.target === waitlistModal) {
        closeModal();
      }
    });
  }

  if (waitlistForm) {
    waitlistForm.addEventListener("submit", (e) => {
      e.preventDefault();
      
      const email = waitlistEmailInput.value;
      if (!email) return;
      
      const originalBtnText = waitlistSubmitBtn.textContent;
      waitlistSubmitBtn.disabled = true;
      waitlistSubmitBtn.textContent = "Joining...";
      waitlistSubmitBtn.style.opacity = "0.7";
      
      fetch("https://formsubmit.co/ajax/neuronlabs768@gmail.com", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({
          email: email,
          _subject: "New NEURON Cloud Platform Waitlist Signup!",
          _message: `Developer email: ${email} registered for the NEURON SaaS Cloud Beta.`
        })
      })
      .then(response => response.json())
      .then(data => {
        waitlistSuccessEmail.textContent = email;
        waitlistFormContainer.style.display = "none";
        waitlistSuccessContainer.style.display = "block";
      })
      .catch(error => {
        console.error("Waitlist submission error:", error);
        waitlistSuccessEmail.textContent = email;
        waitlistFormContainer.style.display = "none";
        waitlistSuccessContainer.style.display = "block";
      })
      .finally(() => {
        waitlistSubmitBtn.disabled = false;
        waitlistSubmitBtn.textContent = originalBtnText;
        waitlistSubmitBtn.style.opacity = "1";
      });
    });
  }

  // 14. Scroll Reveal Observer
  const revealElements = document.querySelectorAll(".reveal");
  if (revealElements.length > 0 && 'IntersectionObserver' in window) {
    const revealObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add("revealed");
          observer.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.1,
      rootMargin: "0px 0px -40px 0px"
    });
    revealElements.forEach(el => revealObserver.observe(el));
  } else {
    revealElements.forEach(el => el.classList.add("revealed"));
  }

  // 15. Premium Cursor-Tracking Hover Effect
  const cards = document.querySelectorAll(".feature-card, .pricing-card, .benchmark-card, .cta-card");
  cards.forEach(card => {
    card.addEventListener("mousemove", (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      card.style.setProperty("--mouse-x", `${x}px`);
      card.style.setProperty("--mouse-y", `${y}px`);
    });
  });

  // 16. Stats Count-Up Animation
  const statsElements = document.querySelectorAll(".hero-stat-value");
  if (statsElements.length > 0 && 'IntersectionObserver' in window) {
    const statsObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const stat = entry.target;
          const target = parseInt(stat.getAttribute("data-target"));
          if (isNaN(target)) return;
          if (target === 0) {
            stat.textContent = "0";
            observer.unobserve(stat);
            return;
          }
          let current = 0;
          const duration = 1500;
          const steps = 50;
          const stepTime = duration / steps;
          const increment = target / steps;
          
          let step = 0;
          const timer = setInterval(() => {
            current += increment;
            step++;
            if (step >= steps) {
              clearInterval(timer);
              if (target >= 100000) {
                stat.textContent = "100k+";
              } else {
                stat.textContent = Math.round(target).toString();
              }
            } else {
              if (target >= 100000) {
                stat.textContent = Math.round(current / 1000) + "k+";
              } else {
                stat.textContent = Math.round(current).toString();
              }
            }
          }, stepTime);
          observer.unobserve(stat);
        }
      });
    }, { threshold: 0.2 });
    statsElements.forEach(el => statsObserver.observe(el));
  } else {
    statsElements.forEach(el => {
      const target = el.getAttribute("data-target");
      el.textContent = target === "100000" ? "100k+" : target;
    });
  }

  // Initialize playground on startup
  initPlayground();

  // CSS Spin keyframes injection for runner loading icon
  const style = document.createElement('style');
  style.innerHTML = `
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  `;
  document.head.appendChild(style);
});
