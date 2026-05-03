import React from "react";
import { Route, Routes } from "react-router-dom";
import Layout from "@/components/Layout";
import Dashboard from "@/pages/Dashboard";
import Chat from "@/pages/Chat";
import Research from "@/pages/Research";
import DCFLab from "@/pages/DCFLab";
import Comps from "@/pages/Comps";
import Screener from "@/pages/Screener";
import PortfolioBuilder from "@/pages/PortfolioBuilder";
import Macro from "@/pages/Macro";
import Settings from "@/pages/Settings";
import TrackRecord from "@/pages/TrackRecord";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/research" element={<Research />} />
        <Route path="/dcf" element={<DCFLab />} />
        <Route path="/comps" element={<Comps />} />
        <Route path="/screener" element={<Screener />} />
        <Route path="/portfolio" element={<PortfolioBuilder />} />
        <Route path="/macro" element={<Macro />} />
        <Route path="/track-record" element={<TrackRecord />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="*" element={<Dashboard />} />
      </Route>
    </Routes>
  );
}
