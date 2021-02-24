fn foo(a: int) {
    loop {
        if (a < 5) {
            break;
        }
        a = (((a - 1 + 3) / 4) << 2) % 5;
    }
}