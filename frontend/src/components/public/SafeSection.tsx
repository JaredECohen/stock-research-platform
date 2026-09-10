import React from "react";

/**
 * Error boundary for sample renderers. A sample payload is a stored,
 * stripped copy of a memo written by an older pipeline version, so a
 * field the app's components assume can be missing. One bad section must
 * not blank the whole landing page; it becomes a labelled placeholder.
 */
interface Props {
  label: string;
  children: React.ReactNode;
}

interface State {
  failed: boolean;
}

export default class SafeSection extends React.Component<Props, State> {
  state: State = { failed: false };

  static getDerivedStateFromError(): State {
    return { failed: true };
  }

  componentDidCatch(): void {
    // Nothing to log: no error text ever leaves the browser from a public
    // page, and the visitor sees the placeholder below.
  }

  render(): React.ReactNode {
    if (this.state.failed) {
      return (
        <div className="card-tight text-sm text-slate-400" role="note" data-testid="section-failed">
          {this.props.label} could not be displayed for this sample.
        </div>
      );
    }
    return this.props.children;
  }
}
