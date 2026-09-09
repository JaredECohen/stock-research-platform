import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { AuthProvider } from "./auth/AuthProvider";
import { ConfigProvider } from "./auth/ConfigProvider";
import "./index.css";

// ConfigProvider fetches /api/public/config (2s timeout, safe defaults);
// AuthProvider then mounts Clerk, the stub, or nothing. Both sit inside the
// router because sign-in needs to navigate.
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <ConfigProvider>
        <AuthProvider>
          <App />
        </AuthProvider>
      </ConfigProvider>
    </BrowserRouter>
  </React.StrictMode>,
);
