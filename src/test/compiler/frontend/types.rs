struct foo {
    a: [int],
    b: [int; 8],
}

fn bar(a: [int], b: [int]) {
    a[0] = b[0] + 5;
}