import { describe, expect, it } from "vitest";
import { batchRows, rowCount } from "./hooks";

describe("batchRows", () => {
  it("labels test images of a torch batch", () => {
    expect(batchRows({ indices: "4,7" })).toEqual({ noun: "image", labels: ["#4", "#7"], images: [4, 7] });
  });

  it("labels the queries of one VNN-LIB property", () => {
    expect(batchRows({ vnnlib: "vnnlib/instance_1.vnnlib", queries: 3, query_rows: 3 }))
      .toEqual({ noun: "query", labels: ["q1", "q2", "q3"], images: null });
  });

  it("keeps a single-query property unbatched", () => {
    expect(batchRows({ vnnlib: "vnnlib/p.vnnlib", queries: 1, query_rows: 1 })).toBeNull();
  });

  it("reads models built with only the first query as one row", () => {
    expect(batchRows({ vnnlib: "vnnlib/instance_1.vnnlib", queries: 38 })).toBeNull();
  });

  it("expands the queries of each property in a batch, property by property", () => {
    const rows = batchRows({ vnnlibs: "vnnlib/a.vnnlib,vnnlib/b.vnnlib", query_counts: "1,2", query_rows: 3 });
    expect(rows).toEqual({ noun: "query", labels: ["a", "b q1", "b q2"], images: null });
  });

  it("keeps property labels when every property of a batch has one query", () => {
    expect(batchRows({ vnnlibs: "vnnlib/a.vnnlib,vnnlib/b.vnnlib", query_counts: "1,1" }))
      .toEqual({ noun: "property", labels: ["a", "b"], images: null });
  });

  it("falls back to property labels when query counts do not match the properties", () => {
    for (const query_counts of ["2", "1,x", "0,1"]) {
      expect(batchRows({ vnnlibs: "vnnlib/a.vnnlib,vnnlib/b.vnnlib", query_counts }))
        .toEqual({ noun: "property", labels: ["a", "b"], images: null });
    }
  });
});

describe("rowCount", () => {
  it("uses the batch noun's plural", () => {
    expect(rowCount(38, "query")).toBe("38 queries");
    expect(rowCount(1, "query")).toBe("1 query");
    expect(rowCount(2, "property")).toBe("2 properties");
    expect(rowCount(3, "image")).toBe("3 images");
    expect(rowCount(2, "row")).toBe("2 rows");
  });
});
