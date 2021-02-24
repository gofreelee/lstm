struct foo {
	a: int,
	b: int,
}

fn bar (p: [foo]) {
    p[0].a = p[0].b + 5;
}