import { useState } from "react";
import type { ConsoleTransport, EnvironmentPackageForm, EnvironmentRegistration } from "../api/transport";
import { EXECUTION_MODES, REQUIRED_PACKAGE_FIELDS, EMPTY_PACKAGE_FORM, formToRegistration } from "../api/transport";
import type { EnvironmentPackageSummary } from "../api/types";
import { StatusBadge } from "../components/StatusBadge";
import { Banner, EmptyState, LoadingState } from "../components/ui";

export function RegistryView({
  transport,
  environments,
  loading,
  onRegistered,
  onActionError,
}: {
  transport: ConsoleTransport;
  environments: EnvironmentPackageSummary[];
  loading: boolean;
  onRegistered: (environment: EnvironmentPackageSummary) => void;
  onActionError: (message: string, correlationId?: string | null) => void;
}) {
  const [form, setForm] = useState<EnvironmentPackageForm>(EMPTY_PACKAGE_FORM);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [submitState, setSubmitState] = useState<"idle" | "checking" | "ok" | "rejected" | "registering" | "registered">("idle");

  const update = (key: keyof EnvironmentPackageForm) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    // preserve user's input on validation failure (SPEC console requirement)
    setForm((f) => ({ ...f, [key]: e.target.value }));
    setFieldErrors((prev) => {
      if (!prev[key]) return prev;
      const next = { ...prev };
      delete next[key];
      return next;
    });
  };

  const toggleMode = (mode: EnvironmentRegistration["executionModes"][number]) => {
    setForm((f) => ({
      ...f,
      executionModes: f.executionModes.includes(mode)
        ? f.executionModes.filter((m) => m !== mode)
        : [...f.executionModes, mode],
    }));
    setFieldErrors((prev) => {
      const next = { ...prev };
      delete next.executionModes;
      return next;
    });
  };

  const fieldErrorsFrom = (missing: string[]): Record<string, string> => {
    const errors: Record<string, string> = {};
    for (const key of missing) {
      const label = REQUIRED_PACKAGE_FIELDS.find((f) => f.key === key)?.label ?? (key === "executionModes" ? "Execution modes" : key);
      errors[key] = `${label} is required or invalid`;
    }
    return errors;
  };

  const validate = async () => {
    setSubmitState("checking");
    const result = await transport.validateEnvironmentPackage(form);
    if (result.ok) {
      setSubmitState("ok");
      setFieldErrors({});
    } else {
      setSubmitState("rejected");
      setFieldErrors(fieldErrorsFrom(result.missingFields)); // form values are NOT reset
    }
  };

  const register = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitState("registering");
    setFieldErrors({});
    try {
      const manifest = formToRegistration(form);
      const environment = await transport.registerEnvironment(manifest);
      setSubmitState("registered");
      onRegistered(environment);
      setForm(EMPTY_PACKAGE_FORM);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Registration failed";
      const correlationId = (error as { correlationId?: string } | null)?.correlationId ?? null;
      onActionError(`Environment registration failed: ${message}`, correlationId);
      // keep the operator's input for correction
      setSubmitState("rejected");
    }
  };

  if (loading) return <LoadingState label="Loading environment registry…" />;

  return (
    <section aria-labelledby="registry-heading" className="grid gap-6 lg:grid-cols-2">
      <div>
        <h2 id="registry-heading" className="text-base font-semibold text-slate-100">
          Registered environments
        </h2>
        {environments.length === 0 ? (
          <div className="mt-3">
            <EmptyState title="No environments registered">
              An environment package must declare: tool schemas, documentation with classifications, declared
              execution modes, a policy reference, a trusted evaluator reference, and a reset fixture reference.
              Validation rejects packages missing any of these before a run starts.
            </EmptyState>
          </div>
        ) : (
          <ul className="mt-3 space-y-3">
            {environments.map((env) => (
              <li key={env.environmentId} className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="font-medium text-slate-100">
                    {env.environmentId} <span className="text-slate-500">v{env.version}</span>
                  </p>
                  <StatusBadge status={env.validationState === "valid" ? "validated" : "invalid"} />
                </div>
                <dl className="mt-2 grid grid-cols-1 gap-x-6 gap-y-1 text-[13px] text-slate-400 sm:grid-cols-2">
                  <div className="flex justify-between gap-2">
                    <dt>Tools</dt>
                    <dd className="text-slate-300">{env.toolCount}</dd>
                  </div>
                  <div className="flex justify-between gap-2">
                    <dt>Policy scope</dt>
                    <dd className="text-slate-300">{env.policyScope}</dd>
                  </div>
                  <div className="flex justify-between gap-2">
                    <dt>Evaluator</dt>
                    <dd>
                      <StatusBadge status={env.evaluatorReady ? "ready" : "stale"} />
                    </dd>
                  </div>
                </dl>
                {!env.evaluatorReady && (
                  <div className="mt-2">
                    <Banner tone="warn" title="Evaluator unavailable">
                      Learning activation is blocked. Retry after the evaluator recovers.
                    </Banner>
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div>
        <h2 className="text-base font-semibold text-slate-100">Register an environment package</h2>
        <form onSubmit={register} noValidate className="mt-3 space-y-3 rounded-xl border border-slate-700 bg-slate-900/60 p-4">
          <div className="grid gap-3 sm:grid-cols-2">
            {(["environmentId", "version", "policyRef", "evaluatorRef", "resetRef"] as const).map((key) => {
              const label = REQUIRED_PACKAGE_FIELDS.find((f) => f.key === key)!.label;
              return (
                <div key={key}>
                  <label htmlFor={`pkg-${key}`} className="block text-xs font-medium text-slate-400">
                    {label}
                  </label>
                  <input
                    id={`pkg-${key}`}
                    value={form[key]}
                    onChange={update(key)}
                    aria-invalid={fieldErrors[key] ? true : undefined}
                    aria-describedby={fieldErrors[key] ? `pkg-${key}-error` : undefined}
                    className="mt-1 w-full rounded-md border border-slate-600 bg-slate-800 px-2.5 py-1.5 text-sm text-slate-100 placeholder-slate-500"
                    placeholder={label}
                  />
                  {fieldErrors[key] && (
                    <p id={`pkg-${key}-error`} role="alert" className="mt-1 text-xs text-rose-400">
                      {fieldErrors[key]}
                    </p>
                  )}
                </div>
              );
            })}
          </div>

          <fieldset>
            <legend className="text-xs font-medium text-slate-400">Declared execution modes</legend>
            <div className="mt-1 grid gap-1 sm:grid-cols-2">
              {EXECUTION_MODES.map((mode) => (
                <label key={mode.value} className="flex items-center gap-2 text-[13px] text-slate-300">
                  <input
                    type="checkbox"
                    checked={form.executionModes.includes(mode.value)}
                    onChange={() => toggleMode(mode.value)}
                    className="h-3.5 w-3.5 rounded border-slate-600 bg-slate-800"
                  />
                  <span>{mode.label}</span>
                </label>
              ))}
            </div>
            {fieldErrors.executionModes && (
              <p role="alert" className="mt-1 text-xs text-rose-400">
                {fieldErrors.executionModes}
              </p>
            )}
          </fieldset>

          {([
            ["docs", "Documentation (JSON array: [{id, sha256, classification: \"learner\"|\"operator\"}])", 4],
            ["toolSchemas", "Tool schemas (JSON array: [{name, version, inputSchema, outputSchema, effect}])", 4],
            ["capabilities", "Capability metadata (JSON array of strings)", 2],
          ] as const).map(([key, label, rows]) => (
            <div key={key}>
              <label htmlFor={`pkg-${key}`} className="block text-xs font-medium text-slate-400">
                {label}
              </label>
              <textarea
                id={`pkg-${key}`}
                rows={rows}
                value={form[key]}
                onChange={update(key)}
                aria-invalid={fieldErrors[key] ? true : undefined}
                aria-describedby={fieldErrors[key] ? `pkg-${key}-error` : undefined}
                className="mt-1 w-full rounded-md border border-slate-600 bg-slate-800 px-2.5 py-1.5 font-mono text-xs text-slate-100"
              />
              {fieldErrors[key] && (
                <p id={`pkg-${key}-error`} role="alert" className="mt-1 text-xs text-rose-400">
                  {fieldErrors[key]}
                </p>
              )}
            </div>
          ))}

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => void validate()}
              className="rounded-md border border-slate-600 px-4 py-1.5 text-sm font-medium text-slate-300 hover:bg-slate-800 disabled:opacity-60"
              disabled={submitState === "checking" || submitState === "registering"}
            >
              {submitState === "checking" ? "Validating…" : "Validate package"}
            </button>
            <button
              type="submit"
              className="rounded-md bg-sky-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-60"
              disabled={submitState === "checking" || submitState === "registering"}
            >
              {submitState === "registering" ? "Registering…" : "Register environment"}
            </button>
            {submitState === "rejected" && (
              <p role="alert" className="text-xs text-rose-400">
                Validation failed — your input is preserved below for correction.
              </p>
            )}
            {submitState === "registered" && (
              <p role="status" className="text-xs text-emerald-400">
                ✓ Environment registered.
              </p>
            )}
          </div>
        </form>
      </div>
    </section>
  );
}
