// Queries/mutations for /api/models, /api/models/{id}, /api/models/{id}/rebuild, /api/sources, /api/sources/sample and
// /api/models/{id}/provenance (+ /nearest).
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { InputProvenanceOut } from "../generated/wire";
import { keys } from "../shared/queryClient";
import * as api from "./api";

export const useModels = () => useQuery({ queryKey: keys.models(), queryFn: api.models });
export const useModel = (id: string | null) =>
  useQuery({ queryKey: keys.model(id), queryFn: () => api.model(id!), enabled: id !== null });
export const useSources = () => useQuery({ queryKey: keys.sources(), queryFn: api.sources });
export const useSample = (dataset: string | null, index: number) =>
  useQuery({ queryKey: keys.sample(dataset, index), queryFn: () => api.sample(dataset!, index), enabled: dataset !== null });
export const useProvenance = (id: string | null) =>
  useQuery({ queryKey: keys.provenance(id), queryFn: () => api.provenance(id!), enabled: id !== null });

/** Dataset metadata of one input row (source sentence, class names, source point), or null when there is none. */
export function useRowProvenance(id: string | null, row: number): InputProvenanceOut | null {
  const q = useProvenance(id);
  return q.status === "success" && q.data.status === "available" ? q.data.rows[row] ?? null : null;
}

/** Known examples nearest to a point of the model input space; disabled while point is null. */
export function useNearest(id: string | null, row: number, point: number[] | null) {
  const key = point ? point.map((v) => v.toPrecision(9)).join(",") : "";
  const body = point && point.length > 0
    ? { row, point: [point[0]!, ...point.slice(1)] as [number, ...number[]], k: 3 } : null;
  return useQuery({ queryKey: keys.nearest(id, row, key), queryFn: () => api.nearest(id!, body!),
    enabled: id !== null && body !== null });
}

// What each ACT input row of a batched model is: a test image (torch batch), a VNN-LIB property on one network, or
// one query of a VNN-LIB property (models built with every query as a row carry query_rows; batches add
// query_counts, rows grouped by property in vnnlibs order).
export type BatchRows = { noun: "image" | "property" | "query"; labels: string[]; images: number[] | null };

export function batchRows(p: Record<string, unknown> | undefined): BatchRows | null {
  if (typeof p?.indices === "string") {
    const images = p.indices.split(",").map(Number);
    return { noun: "image", labels: images.map((x) => `#${x}`), images };
  }
  if (typeof p?.vnnlibs === "string") {
    const stems = p.vnnlibs.split(",").map(fileStem);
    const counts = typeof p.query_counts === "string" ? p.query_counts.split(",").map(Number) : null;
    if (!counts || counts.length !== stems.length || !counts.every((n) => Number.isInteger(n) && n >= 1)) {
      return { noun: "property", labels: stems, images: null };
    }
    const labels = stems.flatMap((s, i) => counts[i] === 1 ? [s] : queryLabels(counts[i]!).map((q) => `${s} ${q}`));
    return { noun: counts.some((n) => n > 1) ? "query" : "property", labels, images: null };
  }
  const rows = p?.query_rows;
  if (typeof rows === "number" && Number.isInteger(rows) && rows > 1) return { noun: "query", labels: queryLabels(rows), images: null };
  return null;
}

function queryLabels(n: number): string[] {
  return Array.from({ length: n }, (_, j) => `q${j + 1}`);
}

/** "3 images", "1 property", "38 queries": a row count in the batch's own noun. */
export function rowCount(n: number, noun: BatchRows["noun"] | "row"): string {
  const plural = noun === "property" ? "properties" : noun === "query" ? "queries" : `${noun}s`;
  return `${n} ${n === 1 ? noun : plural}`;
}

export function useBatchRows(modelId: string | null): BatchRows | null {
  const models = useModels();
  return batchRows(models.status === "success" ? models.data.find((m) => m.model_id === modelId)?.params : undefined);
}

export function fileStem(path: string): string {
  return path.split("/").at(-1)!.replace(/\.(onnx|vnnlib)$/, "");
}

/** Name of an instance's network: its ONNX file, or the files of a multi-network property (comma-separated paths). */
export function netStem(onnx: string): string {
  return onnx.split(",").map(fileStem).join(" + ");
}

export function useRebuildModel() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: api.rebuildModel,
    onSuccess: () => client.invalidateQueries({ queryKey: keys.models() }),
  });
}

export function useLoadModel() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: api.loadModel,
    onSuccess: () => client.invalidateQueries({ queryKey: keys.models() }),
  });
}
