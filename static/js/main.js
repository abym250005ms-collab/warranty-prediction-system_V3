(function () {
    const root = document.documentElement;
    const storageKey = "dashboard-theme";
    const themeToggle = document.getElementById("theme-toggle");
    const loadingOverlay = document.getElementById("loading-overlay");
    const healthStatus = document.getElementById("health-status");

    const setTheme = (theme) => {
        root.setAttribute("data-theme", theme);
        localStorage.setItem(storageKey, theme);
        if (themeToggle) {
            themeToggle.textContent = theme === "light" ? "☀️" : "🌙";
        }
    };

    const initializeTheme = () => {
        const savedTheme = localStorage.getItem(storageKey);
        if (savedTheme) {
            setTheme(savedTheme);
            return;
        }
        const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
        setTheme(prefersLight ? "light" : "dark");
    };

    const updateHealth = async () => {
        if (!healthStatus) {
            return;
        }
        try {
            const response = await fetch("/api/health");
            const payload = await response.json();
            if (payload.status === "ok") {
                healthStatus.textContent = "Data status: Healthy";
                healthStatus.style.color = "#22c55e";
            } else {
                healthStatus.textContent = "Data status: Degraded";
                healthStatus.style.color = "#f59e0b";
            }
            console.info("Dashboard health", payload);
        } catch (error) {
            healthStatus.textContent = "Data status: Unavailable";
            healthStatus.style.color = "#ef4444";
            console.error("Health endpoint error", error);
        }
    };

    document.addEventListener("DOMContentLoaded", () => {
        initializeTheme();
        updateHealth();

        if (themeToggle) {
            themeToggle.addEventListener("click", () => {
                const current = root.getAttribute("data-theme") || "dark";
                setTheme(current === "dark" ? "light" : "dark");
            });
        }

        setTimeout(() => {
            if (loadingOverlay) {
                loadingOverlay.classList.add("hidden");
            }
            console.log("Warranty dashboard fully initialized");
        }, 450);
    });
})();
