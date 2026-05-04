import { useEffect } from "react";
import { useLocation } from "react-router-dom";
import { logEvent } from "@/lib/logger";

/**
 * Wave 8G — emits a `route` event every time the URL changes.
 * Mounted once near the app root. Pure side-effect; renders nothing.
 */
export default function RouteTracker() {
  const loc = useLocation();
  useEffect(() => {
    logEvent({
      kind: "route",
      path: loc.pathname + loc.search,
      payload: {
        pathname: loc.pathname,
        search: loc.search,
        state: loc.state ?? null,
      },
    });
  }, [loc.pathname, loc.search]);
  return null;
}
